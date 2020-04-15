"""
TO-DO : 테스트 진행하면서 단계별로 구현할 것.
  1. Training Logging 관리
  2. 모델 / 한글 processing 등에서 사용되는 Configuration의 관리 방법 고민.
"""
import os
import argparse

import numpy as np

from utils import text_process_aihub as aihub
from utils.datasets import AIHubDataSets

from net.model_ctc import CTC_model
# TBD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    ########################
    # Prepare Dataset      #
    ########################
    train_set = AIHubDataSets()
    valid_set = AIHubDataSets()
    test_set  = AIHubDataSets()
    
    train_set.load_datasets(pickle_dir='./data_pickle/data.train.pickle')
    valid_set.load_datasets(pickle_dir='./data_pickle/data.valid.pickle')
    test_set.load_datasets(pickle_dir='./data_pickle/data.test.pickle')

    # f_list_train, l_list_train, f_list_train = train_set.next_batch(batch_size=16)
    # f_list_valid, l_list_valid, f_list_valid = valid_set.next_batch(batch_size=16)
    # f_list_test, l_list_test, f_list_test    = test_set.next_batch(batch_size=16)
    
    #########################
    # Set Model Parameter   #
    #########################

    NUM_CLASSES = len(aihub.HANGULS) + len(aihub.META_TAGS) + 1  # 75, including space and extra meta tags
    FEATURE_DIM = 39     # mfcc feature dimension (frequency axis size)
    CTC_LABELS  = None   # TBD, for decode hypothesis

    MODEL_NAME  = "KOREAN_CTC_20200413"

    print("  * Number of classes : ", NUM_CLASSES)
    print("  * MFCC Feature dim. : ", FEATURE_DIM)
    print("  * CTC Labels        : ", CTC_LABELS)
    print("  * MODEL NAME        : ", MODEL_NAME)

    ##########################
    # Build Model graph      #
    ##########################
    model = CTC_model(NUM_CLASSES, FEATURE_DIM, CTC_LABELS, MODEL_NAME)

    ##########################
    # Training               #
    ##########################
    LEARNING_RATE       = 1e-3
    BATCH_SIZE          = 16
    NUM_EPOCHS          = 100
    
    VALIDATION_LOG_DIR  = ""
    TENSORBOARD_LOG_DIR = ""
    MODEL_SAVE_DIR      = ""
    MODEL_SAVE_STEP     = 1   # unit : epochs
    SUMMARY_LOG_DIR     = ""
    FINAL_MODEL_DIR     = None

    model.train((train_set, valid_set, test_set), NUM_EPOCHS, BATCH_SIZE)

    ##########################
    # Feeding datasets       #
    ##########################
    # f_list_train, l_list_train, f_list_train = train_set.next_batch(batch_size=16)
    # f_list_valid, l_list_valid, f_list_valid = valid_set.next_batch(batch_size=16)
    # f_list_test, l_list_test, f_list_test    = test_set.next_batch(batch_size=16)

    """
    def transform_for_feed(features, labels):
        # labels   = data[0]
        # features = data[1]

        ## For Features
        feat_len = features[0].shape[1]

        input_len_feed = np.asarray(list(map(len, features)))
        inputs_feed    = np.zeros((len(features), max(input_len_feed), feat_len), np.float32)

        for i, l in enumerate(input_len_feed):
            inputs_feed[i, :l, :] = features[i]

        ## For labels
        indices = []
        values  = []
        for index, label in enumerate(labels):
            indices.extend(zip([index] * len(label), range(len(label))))
            values.extend(label)
            # values.extend(map(lambda l: l, label))
        indices = np.asarray(indices, dtype=np.int64)
        values  = np.asarray(values, dtype=np.int32)
        shape   = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
        
        ## Return
        return inputs_feed, input_len_feed, indices, values, shape

    _inputs, _input_len, _indices, _values, _shape = transform_for_feed(f_list_train, l_list_train)

    print(l_list_train[0])
    print(_indices.shape)
    print(np.asarray(_indices).max(0))
    print(np.asarray(_indices))
    print(_indices)
    """
    
