import numpy as np
import scipy.io.wavfile


from math import floor
from numpy.random import randn 
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann


def add_overlapping_blocks(B, R = 0.5):
    [count, nw] = B.shape
    step = floor(nw * R)

    n = (count-1) * step + nw

    x = np.zeros((n, ))

    for i in range(count):
        offset = i * step
        x[offset : nw + offset] += B[i, :]

    return x

def run_source_filter(a, g, block_size):
    src = np.sqrt(g)*randn(block_size, 1) # noise
    
    b = np.concatenate([np.array([-1]), a])
    
    x_hat = lfilter([1], b.T, src.T).T 
    return np.squeeze(x_hat)

def lpc_decode(A, G, w, lowcut = 0):
    [ne, n] = G.shape
    nw = len(w)
    [p, _] = A.shape

    B_hat = np.zeros((n, nw))

    for i in range(n):
        B_hat[i,:] = run_source_filter(A[:, i], G[:, i], nw)

    # recover signal from blocks
    x_hat = add_overlapping_blocks(B_hat);
        
    return x_hat

def lpc_to_wav(A, G, f):
    sample_rate = 8000
    w = hann(floor(0.03*sample_rate), False)
    xhat = lpc_decode(A, G, w)

    scipy.io.wavfile.write("{}.wav".format(f), 8000, xhat)
    print("Written to: {}.wav".format(f))

X = np.load('./data/train_np/89_3.npy')

print(X.shape)

A = X[:100]
G = np.expand_dims(X[100], axis=0)
lpc_to_wav(A, G, 'TEST')
