import os
from numpy import frombuffer
from scipy.io import wavfile


def load_wav(path):
    if os.path.exists(path):
        sr, buff = wavfile.read(path)
        return sr, buff
    else:
        return None, None


def load_pcm(path, endian='int16'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            buff = f.read()
        pcm_data = frombuffer(buff, dtype=endian)
    else:
        pcm_data = None
    return pcm_data


def load_audio(path, sr=16000, endian='int16'):
    if not os.path.exists(path):
        return None, None

    ret = (None, None)
    ext_type = path.split('.')[-1]
    if ext_type == 'wav':
        ret = load_wav(path)
    elif ext_type == 'pcm':
        ret = (sr, load_pcm(path, endian))
    
    return ret


def chop_audio(buff, L=16000):
    if L and len(buff) > L:
        beg = np.random.randint(0, len(buff) - L)
        return buff[beg: beg+L]
    else:
        return buff

