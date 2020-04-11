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


def get_transcript(fname, trans_ext='.txt', ENCODING='euc-kr'):
    txt = ""
    try:
        with open(fname + trans_ext, 'r', encoding=ENCODING) as f:
            txt = f.read().strip()
    except Exception as e:
        print("ERROR: {}".format(e))
    return txt


def get_audiobuff(fname, audio_ext='.pcm', ENDIAN='int16'):
    buff = None
    try:
        with open(fname + audio_ext, 'rb') as f:
            buff = f.read()
        buff = np.frombuffer(buff, dtype=ENDIAN)
    except Exception as e:
        print("ERROR: {}".format(e))
    return buff


def main(args):
    dataset    = args.dataset
    datadir    = args.datadir
    audio_ext  = args.audio_ext
    trans_ext  = args.trans_ext
    endian     = args.endian
    encoding   = args.txt_encoding
    samplerate = args.samplerate

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
        prons, labels = aihub.get_prons_and_labels_from_file(f, rule_in, rule_out, df_korSym, encoding, True, True)
        feature       = transform_mfcc_from_file(f + audio_ext, endian='int16', sr=16000)

        label_list.append(labels)
        feature_list.append(feature)

        # print(" >> File : ", f)
        # print("    ", prons)
        # print("    ", labels)
        # print("    feature shape : ", feature.shape)
    time.sleep(1)
    print("  Data preparation done.")
    print()
    idxs = list(range(len(label_list)))
    shuffle(idxs)
    
    file_list_sf    = [file_list[i] for i in idxs]
    label_list_sf   = [label_list[i] for i in idxs]
    feature_list_sf = [feature_list[i] for i in idxs]

    data = (file_list_sf, feature_list_sf, label_list_sf)
    with open(args.outdir, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
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
    parser.add_argument('--outdir', required=True, type=str, help="dir where preprocessed data will be save")

    args = parser.parse_args()

    print("Args : ", args)

    main(args)

