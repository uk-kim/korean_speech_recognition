import re


def normStep1(txt):
    # (0.1프로)/(영 점 일 프로) or (3G)/(쓰리 쥐) or (1시)/(한 시) 등과 같은
    # (실제 표기)/(발음 표기) 형태의 데이터를 발음 표기의 데이터로 치환하는 함수
    args = re.findall("\(.+?\)/\(.+?\)", txt)
    for arg in args:
        _arg = re.sub("\(.+?\)/", "", arg)
        _arg = re.sub("[\(\)]", "", _arg)
        txt = txt.replace(arg, _arg)
    return txt


def normStep2(txt):
    # . , ? 등 특수문자 제거
    txt = re.sub("[.,?]", "", txt)
    return txt


def normStep3(txt):
    # noise symbol을 [NOISE:o], [NOIST:b] 등과 같이 변환
    symbol_list = re.findall("[a-z]/", txt)
    for symbol in symbol_list:
        _symbol = symbol.replace('/', '')
        new_symbol = "[NOISE:{}]".format(_symbol)
        txt = txt.replace(symbol, new_symbol)
    return txt


def normStep4(txt):
    # +, *와 같은 불분명한 symbol에 대한 전처리
    txt = re.sub("[\+]", "[AMBIG:DUP]", txt)
    txt = re.sub("[\*]", "[AMBIG:UNK]", txt)
    return txt


def normStep5(txt):
    # 뭐/ 아/ 와 같은 간투어 처리
    for _txt in txt.split():
        symbol_list = re.findall("[가-힣].*?/", _txt)
        for symbol in symbol_list:
            new_symbol = symbol.replace('/', '[GANTU]')
            txt = txt.replace(symbol, new_symbol)
    return txt


def normalize(txt):
    # step1 : to pronunciation
    txt = normStep1(txt)
    # step2 : erase special symbol
    txt = normStep2(txt)
    # step3 : noise tagging
    txt = normStep3(txt)
    # step4 : ambiguous symbol tagging
    txt = normStep4(txt)
    # step5 : tagging for GANTU symbol
    txt = normStep5(txt)
    return txt
