from python_speech_features import mfcc, delta
from utils.audio import load_audio, chop_audio
from numpy import concatenate


def transform_mfcc(buff, sr=16000, win_size=0.025, win_step=0.01, \
                    num_cep=13, nfilt=26, preemph=0.97, appendEnergy=True):
    mfcc_feat = mfcc(buff, samplerate=sr, winlen=win_size, numcep=num_cep, nfilt=nfilt, \
                      preemph=preemph, appendEnergy=appendEnergy)
    d_mfcc = delta(mfcc_feat, 2)
    a_mfcc = delta(d_mfcc, 2)

    out = concatenate([mfcc_feat, d_mfcc, a_mfcc], axis=1)
    return out

def transform_mfcc_from_file(fname, endian='in16', sr=16000, L=None, win_size=0.025, win_step=0.01, \
                              num_cep=13, nfilt=26, preemph=0.97, appendEnergy=True):
    sr, buff = load_audio(fname, sr=sr, endian=endian)
    
    if L:
        buff     = chop_audio(buff, L)
    feat = transform_mfcc(buff, sr, win_size, win_step, num_cep, nfilt, preemph, appendEnergy)
    return feat

