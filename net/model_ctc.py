"""
TO-DO : 아래 각 단계는 테스트를 동반하며 진행할 것.
  1. data feeding     : sparse_placeholder에 유의해서 입력 출력 포멧을 잘 관리하기.
  2. decoding         : hypotheses 결과에 대해서 한글로 변환하는 과정
  3. training         : 1, 2를 유의해서 학습하는 과정
  4. training(detail) : latest model restore, tensorboard, summary, logging 등 3번 고도화
"""

import tensorflow as tf


class CTC_model:
    def __init__(self, num_classes, feat_len, ctc_labels, model_name):
        """
        arguments
            num_classes : number of phone classes
            feat_len    : input feature's dimension
            ctc_labels  : phone class id and name mapping information
            model_name  : name scope of this model
        """
        self.NUM_CLASSES  = num_classes  # 1(space) + characters
        self.FEAT_LEN     = feat_len
        self.CTC_LABELS   = ctc_labels
        self.MODEL_NAME   = model_name

        #self.SEQ_LEN = 20
        #self.BATCH_SIZE   = 16
        
        """
        Shape of placeholder tensor
         - self.inputs    : [Batch size, Sequence length, Feature length]
         - self.input_len : [Batch size]
         - self.output
        """
        self.inputs    = tf.placeholder(tf.float32, [None, None, self.FEAT_LEN])
        # self.inputs    = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.SEQ_LEN, self.FEAT_LEN])
        self.input_len = tf.placeholder(tf.int32, [None])
        #self.input_len = tf.placeholder(tf.int32, [self.BATCH_SIZE])
        self.output    = tf.sparse_placeholder(tf.int32)
        self.is_train  = tf.placeholder(tf.bool)

        self.__build__()
        1
    
    def __build__(self):
        self.ctc_loss, self.decoded, self.ler = self.__encoder__(self.inputs, 8)

        
        # return self.ctc_loss, self.decoded, self.ler
    
    def __encoder__(self, net_in, n_filt=5, temporal_stride=1, scope_name=None, output_dropout_prob=0.3, state_dropout_prob=0.3):
        h = net_in

        # Convolutional layer before RNN
        with tf.variable_scope('conv_layer') as scope:
            output_keep_prob = tf.cond(self.is_train, lambda: 1.0 - output_dropout_prob, lambda: 1.0)
            state_keep_prob  = tf.cond(self.is_train, lambda: 1.0 - state_dropout_prob, lambda: 1.0)

            kernel = tf.get_variable('weights',
                                    shape=[10, 1, 1, n_filt], 
                                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                               mode='FAN_IN',
                                                                                               uniform=False,
                                                                                               seed=None,
                                                                                               dtype=tf.float32),
                                    dtype=tf.float32)
            # [batch, seq_len, feat_dim, 1]
            h = tf.expand_dims(h, dim=-1)
            feats = h
            # [batch, seq_len, feat_dim, n_filt]
            h = tf.nn.conv2d(h, kernel, [1, temporal_stride, 1, 1], padding='SAME')

            biases = tf.get_variable('biases', [n_filt],
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.float32)
            h = tf.nn.bias_add(h, biases)
            h = tf.nn.relu(h, name = scope.name)
            h = tf.nn.dropout(h, output_keep_prob)
        # Recurrent layers
        with tf.variable_scope('rnn_layer') as scope:
            # [batch, seq_len, feat_dim * n_filt]
            h = tf.reshape(h, [tf.shape(self.inputs)[0], -1, self.FEAT_LEN * n_filt])

            # hidden size : 128, stacked 2 layer LSTM
            cells = [tf.contrib.rnn.LSTMCell(128) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(cell,
                                                   output_keep_prob=output_keep_prob,
                                                   state_keep_prob=state_keep_prob,
                                                   variational_recurrent=True,
                                                   dtype=tf.float32) for cell in cells]
            h, _ = tf.nn.bidirectional_dynamic_rnn(*cells, h, self.input_len, dtype=tf.float32)
            h = tf.reshape(tf.concat(h, 2), [-1, 2 * 128])

        # FC layers
        with tf.variable_scope('fc_layer') as scope:
            w = tf.Variable(tf.truncated_normal([2 * 128, self.NUM_CLASSES], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[self.NUM_CLASSES]))

            matmul = tf.matmul(h, w) + b
            logits = tf.transpose(tf.reshape(tf.matmul(h, w) + b, [tf.shape(feats)[0], -1, self.NUM_CLASSES]), (1, 0, 2))
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, self.input_len, beam_width=128)

        with tf.name_scope("Loss"):
            ##### targets, seq_len
            ctc = tf.reduce_mean(tf.nn.ctc_loss(self.output, logits, self.input_len, ignore_longer_outputs_than_inputs=True))

        with tf.name_scope("ler"):
            ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.output))
        return ctc, decoded, ler

    def __build_op__(self, config, learning_rate):
        if not self.sess:
            self.sess = tf.Session(config=config)

        global_step = tf.train.get_or_create_global_step()
        self.op = tf.train.AdamOptimizer(learning_rate).minimize(self.ctc_loss, global_step, name="train_op")
        

    def train(self, dataset, num_epochs, model_path='./model', log_dir='./logs', learning_rate=0.001):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                allow_soft_placement=True,
                                log_device_placement=True)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        writers = {phase: tf.summary.FileWriter(path.join(log_dir, phase)) for phase in ["train_sen", "valid_sen", "test_sen"]}
        
        self.__build_op__(config, learning_rate)
        try:
            saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
            global_step = tf.train.get_global_step()
            train_op    = tf.get_default_graph().get_tensor_by_name("train_op:0")
            step        = self.sess.run(global_step)
        except ValueError:
            self.sess.run(tf.global_variables_initializer())
            step = 0

        train_set = dataset[0]
        vaild_set = dataset[1]
        test_set  = dataset[2]





    def predict(self):
        1

    def feed(self):
        1

if __name__ == "__main__":
    num_classes = 80
    feat_len = 39
    n_features = 30
    ctc_labels = None
    model_name = None
    
    print("="*100)
    print(" num_classes == ", num_classes)
    print(" feat_len    == ", feat_len)

    model = CTC_model(num_classes, feat_len, ctc_labels, model_name)


