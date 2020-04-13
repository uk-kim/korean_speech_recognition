"""
TO-DO : 테스트 진행하면서 단계별로 구현할 것.
  1. Training Logging 관리
  2. 모델 / 한글 processing 등에서 사용되는 Configuration의 관리 방법 고민.
"""
import os
import argparse

from utils import text_process_aihub as aihub
from utils.datasets import AIHubDataSets

from net.model_ctc import CTC_model
# TBD



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

    NUM_CLASSES = len(aihub.HANGULS) + len(aihub.META_TAGS)  # 75, including space and extra meta tags
    FEATURE_DIM = 39     # mfcc feature dimension (frequency axis size)
    CTC_LABELS  = None   # TBD, for decode hypothesis

    MODEL_NAME  = "KOREAN_CTC_20200413"

    print("Number of classes : ", NUM_CLASSES)
    print("MFCC Feature dim. : ", FEATURE_DIM)
    print("CTC Labels        : ", CTC_LABELS)
    print("MODEL NAME        : ", MODEL_NAME)

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


