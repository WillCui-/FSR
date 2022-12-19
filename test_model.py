import numpy as np
import nussl
import scipy.io.wavfile
import time
import torch

from math import floor
from numpy.random import randn 
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann

# LPC code

def create_overlapping_blocks(x, w, R = 0.5):
    n = len(x)
    nw = len(w)
    step = floor(nw * (1 - R))
    nb = floor((n - nw) / step) + 1

    B = np.zeros((nb, nw))

    
    for i in range(nb):
        offset = i * step
        B[i, :] = w * x[offset : nw + offset]
        
    return B

def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])
    
    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i 
        X[i, :] = xz[offset : offset + p]
    return X

def solve_lpc(x, p, ii):
    b = x[1:].T
        
    X = make_matrix_X(x, p)
    
    a = np.linalg.lstsq(X, b, rcond=None)[0]

    e = b.T - np.dot(X, a)
    g = np.var(e)

    return [a, g]


def lpc_encode(x, p, w):
    B = create_overlapping_blocks(x, w)
    
    [nb, nw] = B.shape

    A = np.zeros((p, nb))
    G = np.zeros((1, nb))

    for i in range(nb):
        [a, g] = solve_lpc(B[i, :], p, i)
   
        A[:, i] = a
        G[:, i] = g
    
    return [A, G]

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


def wav_to_lpc(file):
    # Open .wav file
    [sample_rate, pcm_data] = scipy.io.wavfile.read(file)
    scipy.io.wavfile.write("example_orig.wav", sample_rate, pcm_data)

    # Turn stereo into mono
    pcm_data = np.max(pcm_data, axis=1)
    scipy.io.wavfile.write("example_max.wav", sample_rate, pcm_data)

    # Normalize audio
    pcm_data = 0.9*pcm_data/max(abs(pcm_data))

    # Resample to 8kHz
    target_sample_rate = 8000
    target_size = int(len(pcm_data) * target_sample_rate / sample_rate)
    pcm_data = resample(pcm_data, target_size) 
    sample_rate = target_sample_rate
    scipy.io.wavfile.write("example_resample.wav", sample_rate, pcm_data)

    w = hann(floor(0.03*sample_rate), False)
    
    print(pcm_data.shape)

    # Encoding .wav
    p = 100 # number of poles
    [A, G] = lpc_encode(pcm_data, p, w)

    return A, G

def lpc_to_wav(A, G, f):
    sample_rate = 8000
    w = hann(floor(0.03*sample_rate), False)
    xhat = lpc_decode(A, G, w)

    print(xhat.shape)

    scipy.io.wavfile.write("{}.wav".format(f), 8000, xhat)
    print("Written to: {}.wav".format(f))

# End LPC code

model_name = './model_190'

reloaded_dict = torch.load(model_name)

model = nussl.ml.SeparationModel(reloaded_dict['config'])
model.load_state_dict(reloaded_dict['state_dict'])

f = './data/test_mod/{}.wav'.format('0_0')

start_time = time.time()

# Calculating LPC

A, G = wav_to_lpc(f)
lpc_input = np.concatenate([A, G])
lpc_input = torch.from_numpy(np.expand_dims(np.expand_dims(lpc_input.T, 0), 3)).float()
print("Model input: ", lpc_input.shape)

item = {'mix_magnitude': lpc_input}

# Inputting to model
with torch.no_grad():
    output = model(item)

print("Model output", output['my_estimates'].shape)

result = output['my_estimates'].squeeze()
result = result[:, :, -1].numpy().T

print("LPC decode input", result.shape)

A = result[:100]
G = np.expand_dims(result[100], axis=0)
lpc_to_wav(A, G, 'result')

print(time.time() - start_time)



