import os
import re

import sys
sys.path.insert(0, '..')   # To import parent dir's packages

from pandas import DataFrame as dataframe

from utils.text_normalize import normalize
from g2p import g2p
ver_info = sys.version_info

# [rule_in, rule_out] = g2p.readRules(ver_info[0], '../g2p/rulebook.txt')


HANGULS = ['ㅂ', 'ㅍ', 'ㅃ', 'ㄷ', 'ㅌ', 'ㄸ', 'ㄱ', 'ㅋ', 'ㄲ', 'ㅅ', 'ㅆ', 'ㅎ', 'ㅈ', 'ㅊ', 'ㅉ', 'ㅁ', 'ㄴ', 'ㄹ', 'ㅂ', 'ㅍ', 'ㄷ', 'ㅌ', 'ㄱ', 'ㅋ', 'ㄲ', 'ㅅ', 'ㅆ', 'ㅎ', 'ㅈ', 'ㅊ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㄱㅅ', 'ㄴㅈ', 'ㄴㅎ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ', 'ㅂㅅ', 'ㅣ', 'ㅔ', 'ㅐ', 'ㅏ', 'ㅡ', 'ㅓ', 'ㅜ', 'ㅗ', 'ㅖ', 'ㅒ', 'ㅑ', 'ㅕ', 'ㅠ', 'ㅛ', 'ㅟ', 'ㅚ', 'ㅙ', 'ㅞ', 'ㅘ', 'ㅝ', 'ㅢ']
SYMBOLS = ['p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr', 'pf', 'ph', 'tf', 'th', 'kf', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'mf', 'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo', 'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we', 'wa', 'wv', 'xi']
META_TAGS = ['[SPACE]', '[AMBIG:DUP]', '[AMBIG:UNK]', '[GANTU]', '[NOISE:b]', '[NOISE:n]', '[NOISE:l]', '[NOISE:o]', '[UNK]']


def get_korean_symbol_dataframe():
    df_korSym = dataframe()
    df_korSym['Hangul'] = META_TAGS + HANGULS
    df_korSym['Symbol'] = META_TAGS + SYMBOLS
    # df_korSym['Hangul'] = HANGULS
    # df_korSym['Symbol'] = SYMBOLS
    # for tag in META_TAGS:
    #       df_korSym.loc[len(df_korSym)] = [tag, tag]

    return df_korSym


def get_transcript(fname, trans_ext='.txt', ENCODING='euc-kr'):
    txt = ""
    try:
        with open(fname + trans_ext, 'r', encoding=ENCODING) as f:
            txt = f.read().strip()
    except Exception as e:
        print("ERROR: {}".format(e))
    return txt


def kor2pron(txt, rule_in, rule_out, special_symbol='|'):
    pron_list = []
    for w in txt.split():
        tag = re.findall('(\[.+?\])', w)
        if tag:
            w_base = w.replace(tag[0], '')
            prons = g2p.graph2prono(w_base, rule_in, rule_out)
            pron_list.append(special_symbol+prons+' '+tag[0])
        else:
            prons = g2p.graph2prono(w, rule_in, rule_out)
            pron_list.append(special_symbol+prons)
    pronun_txt = " ".join(pron_list)
    
    return pronun_txt


def pron2kor(prons, df_korSym, special_symbol='|'):
    kor_list = []
    for word in prons.split(special_symbol):
        for pron in word.split():
            idxs = df_korSym.index[df_korSym['Symbol'] == pron].tolist()
            if idxs:
                hangul = df_korSym.iloc[idxs[0]]['Hangul']
                kor_list.append(hangul)
        kor_list.append(' ')
    
    return "".join(kor_list)


def pron_to_label(pron_txt, df_korSym, with_space=True, with_unk=True):
    """
    Assume pron_txt is '|aa [GANTU] |mm oo nf |s0 oo rr ii ya | [NOISE:b]' from text '아[GANTU] 몬 소리야 [NOISE:b]'.
      (labels : [48, 68, 73, 15, 52, 31, 73, 9, 52, 17, 45, 55, 73, 69])
    Maybe the begin and end of the correspondence audio has some additional sound likes noise, or the other sound.
    
    So, if you use 'unk_flag' as True,
    then, the result label is including [UNK] label both begin and end point of label sequence.
    
    lables with unk_flag: [74, 48, 68, 73, 15, 52, 31, 73, 9, 52, 17, 45, 55, 73, 69, 74])
    """
    pron_txt = pron_txt[1:] if pron_txt.startswith('|') else pron_txt
    if with_space:
        pron_txt = pron_txt.replace('|', '[SPACE] ')
    else:
        pron_txt = pron_txt.replace('|', '')
    
    idx_list = []
    for tag in pron_txt.split():
        idx = df_korSym.index[df_korSym['Symbol'] == tag].tolist()[0]
        idx_list.append(idx)
    
    unk_idx = df_korSym.index[df_korSym['Symbol'] == '[UNK]'].tolist()[0]
    if with_unk:
        idx_list = [unk_idx] + idx_list + [unk_idx]
    return idx_list


def label_to_pron(labels, df_korSym, with_space=True, with_unk=True):
    pron_txt_list = []
    for label in labels:
        pron = df_korSym.loc[label]['Symbol']
        pron_txt_list.append(pron)
    if with_unk:
        pron_txt_list = pron_txt_list[1:-1]
    
    pron_txt = " ".join(pron_txt_list)
    pron_txt = pron_txt.replace('[SPACE] ', '|')
    pron_txt = '|' + pron_txt
    return pron_txt


def get_prons_and_labels_from_text(text, rule_in, rule_out, df_korSym, with_space=True, with_unk=True):
    text_norm = normalize(text)

    prons   = kor2pron(text_norm, rule_in, rule_out)
    labels = pron_to_label(prons, df_korSym, with_space=with_space, with_unk=with_unk)

    return prons, labels


def get_prons_and_labels_from_file(file, rule_in, rule_out, df_korSym, encoding='euc-kr', with_space=True, with_unk=True):
    text = get_transcript(file, ENCODING=encoding)
    prons, labels = get_prons_and_labels_from_text(text, rule_in, rule_out, df_korSym, with_space, with_unk)

    return prons, labels


def get_prons_and_kor_symbols_from_labels(labels, df_korSym, with_space=True, with_unk=True):
    prons_from_label = label_to_pron(labels, df_korSym, with_space, with_unk)
    text_from_prons  = pron2kor(prons_from_label, df_korSym)

    return prons_from_label, text_from_prons
