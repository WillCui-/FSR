# wyc2112

'''
Evaluates a model based on a ground truth .wav stem and an outputted .wav 
stem and calculates signal-to-distortion ratio

This file reads in two different .wav files in pre-defined locations. Then
it uses the museval library to calculate the signal-to-noise ratio
'''

import museval
import numpy as np
import scipy.io.wavfile

orig_file = './eval/source/r.wav'
res_file = './eval/result/r.wav'

[sample_rate, orig_pcm_data] = scipy.io.wavfile.read(orig_file)

orig_pcm_data = np.max(orig_pcm_data, axis=1)

[sample_rate, res_pcm_data] = scipy.io.wavfile.read(res_file)

orig = np.expand_dims(np.expand_dims(orig_pcm_data, 0), 2)
res = np.expand_dims(np.expand_dims(res_pcm_data, 0), 2)

print(orig.shape)

evaluation_result = museval.evaluate(orig, res)  # , sample_rate, sample_rate)

print(evaluation_result)

# print(evaluation_result[0].tolist())

samples = evaluation_result[0].shape[1]

print("Sum: ", np.nanmean(np.ma.masked_invalid(evaluation_result[0])))
print("Sum: ", np.ma.masked_invalid(evaluation_result[0]).sum()/samples)
print("Sum: ", np.nansum(evaluation_result[0])/samples)


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


print(signaltonoise_dB(res.squeeze()).shape)
