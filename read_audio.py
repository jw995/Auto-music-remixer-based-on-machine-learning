import librosa
import numpy as np

N_FFT = 2048

def read_audio_spectum_offset(filename,val):
    val=float(val);
    x, fs = librosa.load(filename, offset=val, duration=10.0)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs

def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))
    return S, fs
