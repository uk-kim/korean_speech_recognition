import tensorflow as tf


class STT_CTC_model:
    def __init__(self, num_classes, feat_len, n_features, ctc_labels, model_name):
        """
        arguments
            num_classes : number of phone classes
            feat_len    : input feature's dimension
            n_feature   : size of convolution layer's channel before rnn
            ctc_labels  : phone class id and name mapping information
            model_name  : name scope of this model
        """
        self.NUM_CLASSES  = num_classes
        self.FEAT_LEN     = feat_len
        self.NUM_FEATURES = n_features
        self.CTC_LABELS   = ctc_labels
        self.MODEL_NAME   = model_name

        self.SEQ_LEN = 20
        self.BATCH_SIZE   = None
        
        """
        Shape of placeholder tensor
         - self.inputs    : [Batch size, Sequence length, Feature length]
         - self.input_len : [Batch size]
         - self.output
        """
        #self.inputs    = tf.placeholder(tf.float32, [None, None, self.FEAT_LEN])
        self.inputs    = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.SEQ_LEN, self.FEAT_LEN])
        #self.input_len = tf.placeholder(tf.int32, [None])
        self.input_len = tf.placeholder(tf.int32, [self.BATCH_SIZE])
        self.output    = tf.sparse_placeholder(tf.int32)
        self.is_train  = tf.placeholder(tf.bool)

        self.__build__()
        1
    
    def __build__(self):
        self.ctc_loss, self.decoded, self.ler = self.__encoder__(self.inputs, 5)

        
        return self.ctc_loss, self.decoded, self.ler
    
    def __encoder__(self, net_in, n_filt=5, temporal_stride=1, scope_name=None, output_dropout_prob=0.3, state_dropout_prob=0.3):
        h = net_in
        print("  >>  net_in   : ", h)

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
            print("  >>  h_expand_dims : ", h)
            feats = h
            # [batch, seq_len, feat_dim, n_filt]
            h = tf.nn.conv2d(h, kernel, [1, temporal_stride, 1, 1], padding='SAME')

            biases = tf.get_variable('biases', [n_filt],
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.float32)
            h = tf.nn.bias_add(h, biases)
            h = tf.nn.relu(h, name = scope.name)
            h = tf.nn.dropout(h, output_keep_prob)
            print("  >>  h_conv : ", h)
        # Recurrent layers
        with tf.variable_scope('rnn_layer') as scope:
            # [batch, seq_len, feat_dim * n_filt]
            #h = tf.reshpae(h, [-1, h.shape[1], feat_dim * n_filt]
            #h = tf.reshape(h, [self.BATCH_SIZE, -1, self.feat_dim * n_filt])
            h = tf.reshape(h, [tf.shape(self.inputs)[0], -1, self.FEAT_LEN * n_filt])
            #h = tf.reshape(h, [self.BATCH_SIZE, -1, self.FEAT_LEN * n_filt])
            print("  >>  h_rnn_in_reshape : ", h)

            # hidden size : 128, stacked 2 layer LSTM
            cells = [tf.contrib.rnn.LSTMCell(128) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(cell,
                                                   output_keep_prob=output_keep_prob,
                                                   state_keep_prob=state_keep_prob,
                                                   variational_recurrent=True,
                                                   dtype=tf.float32) for cell in cells]
            h, _ = tf.nn.bidirectional_dynamic_rnn(*cells, h, self.input_len, dtype=tf.float32)
            #h, _ = tf.nn.bidirectional_dynamic_rnn(*cells, rnn_input, self.input_len, dtype=tf.float32)
            print("  >>  h_rnn out : ", h)
            print("  >> tf.concat(h, 2) : ", tf.concat(h, 2))
            h = tf.reshape(tf.concat(h, 2), [-1, 2 * 128])
            print("  >>  h_rnn_out_rsp  : ", h)

        # FC layers
        with tf.variable_scope('fc_layer') as scope:
            w = tf.Variable(tf.truncated_normal([2 * 128, self.NUM_CLASSES], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[self.NUM_CLASSES]))

            matmul = tf.matmul(h, w) + b
            print("  >> before logits matmul : ", matmul)
            print("  >> tf.shape(feats) : ", tf.shape(feats))
            logits = tf.transpose(tf.reshape(tf.matmul(h, w) + b, [tf.shape(feats)[0], -1, self.NUM_CLASSES]), (1, 0, 2))
            print("  >>  logits  : ", logits)
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, self.input_len, beam_width=128)
            print("  >>  decoded : ", decoded)

        with tf.name_scope("Loss"):
            ##### targets, seq_len
            ctc = tf.reduce_mean(tf.nn.ctc_loss(self.output, logits, self.input_len, ignore_longer_outputs_than_inputs=True))
            print("  >>  ctc loss : ", ctc)

        with tf.name_scope("ler"):
            ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.output))
            print("  >>  ler : ", ler)
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
        
        self.__build_op(config, learning_rate)
        try:
            saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
            global_step = tf.train.get_global_step()
            train_op    = tf.get_default_graph().get_tensor_by_name("train_op:0")
            step        = self.sess.run(global_step)
        except ValueError:
            self.sess.run(tf.global_variables_initializer())
            step = 0




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
    print(" n_features  == ", n_features)

    model = STT_CTC_model(num_classes, feat_len, n_features, ctc_labels, model_name)


