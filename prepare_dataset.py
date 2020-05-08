import os
import argparse

from utils import text_process_aihub as aihub
from utils.transform import transform_mfcc_from_file

from random import shuffle
import pickle
from tqdm import tqdm
import time


def get_file_list(path, audio_ext='.pcm', trans_ext='.txt'):
    if not os.path.exists(path):
        return []
    
    file_list = []
    for _path, _dir, _files in os.walk(path):
        for f in _files:
            if f[-len(audio_ext):] == audio_ext:
                f_name = os.path.join(_path, f[:-len(audio_ext)])
                if os.path.exists(f_name + trans_ext):
                    file_list.append(f_name)
    file_list.sort()
    
    return file_list


def main(args):
    dataset    = args.dataset
    datadir    = args.datadir
    audio_ext  = args.audio_ext
    trans_ext  = args.trans_ext
    endian     = args.endian
    encoding   = args.txt_encoding
    samplerate = args.samplerate
    ratios     = args.split_ratio

    file_list = get_file_list(datadir, audio_ext, trans_ext)
    
    rule_in, rule_out = aihub.g2p.readRules(aihub.ver_info[0], './g2p/rulebook.txt')
    df_korSym = aihub.get_korean_symbol_dataframe()
    
    print(" Audio format.     : {}".format(audio_ext))
    print(" Transcript format : {}".format(trans_ext))
    print(" Number of files   : {}".format(len(file_list)))
    print(" Head of file list")
    for f in file_list[:5]:
        print("   ", f)
    
    print('='*100)
    
    time.sleep(1)

    label_list   = []
    feature_list = []
    for f in tqdm(file_list):
        text, prons, labels = aihub.get_prons_and_labels_from_file(f, rule_in, rule_out, df_korSym, encoding, True, True)
        feature       = transform_mfcc_from_file(f + audio_ext, endian='int16', sr=16000)

        label_list.append(labels)
        feature_list.append(feature)

    time.sleep(1)
    print("  Data preparation done.")
    print()
    idxs = list(range(len(label_list)))
    shuffle(idxs)
    
    file_list_sf    = [file_list[i] for i in idxs]
    label_list_sf   = [label_list[i] for i in idxs]
    feature_list_sf = [feature_list[i] for i in idxs]

    #### Split dataset into {train / valid / test} sets.
    ratios = [float(rate) for rate in ratios.split(':')]

    n_total_files = len(file_list)
    n_train = int(n_total_files * ratios[0])
    n_valid = int(n_total_files * ratios[1])
    n_test  = n_total_files - n_train - n_valid
    
    data_train = (file_list_sf[:n_train], feature_list_sf[:n_train], label_list_sf[:n_train])
    data_valid = (file_list_sf[n_train:-n_test], feature_list_sf[n_train:-n_test], label_list_sf[n_train:-n_test])
    data_test  = (file_list_sf[-n_test:], feature_list_sf[-n_test:], label_list_sf[-n_test:])

    with open(os.path.join(args.outdir, 'data.train.pickle'), 'wb') as f:
        pickle.dump(data_train, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.outdir, 'data.valid.pickle'), 'wb') as f:
        pickle.dump(data_valid, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.outdir, 'data.test.pickle'), 'wb') as f:
        pickle.dump(data_test, f, pickle.HIGHEST_PROTOCOL)

    print("  Data saved in {}".format(args.outdir))
    print()

    print(" Done.!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script is for prepration with your own dataset.')
    parser.add_argument('--dataset', default='aihub', type=str,
                        help='select one dataset from [aihub, ... ]')
    parser.add_argument('--datadir', required=True, type=str, help='dir where dataset is in')
    parser.add_argument('--audio_ext', default='.pcm', type=str)
    parser.add_argument('--trans_ext', default='.txt', type=str)
    parser.add_argument('--endian', default='int16', type=str)
    parser.add_argument('--txt_encoding', default='euc-kr', type=str)
    parser.add_argument('--samplerate', default=16000, type=int)
    parser.add_argument('--outdir', required=False, default='./', type=str, help="dir where preprocessed data will be save")
    parser.add_argument('--split_ratio', required=False, type=str, \
                        default='0.7:0.2:0.1', help="dir where preprocessed data will be save")

    args = parser.parse_args()

    print("Args : ", args)

    main(args)

