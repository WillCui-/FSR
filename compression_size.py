# wyc2112

'''
Calculates the compression size of LPC. Reads in a .wav file and a LPC coefficieent file
and compares them to each other by printing out their shapes.
'''

print("Loading...")
f = './data/test_mod/{}.wav'.format('0_0')
[_, pcm_data] = scipy.io.wavfile.read(f)
print(pcm_data.shape)

A, G = wav_to_lpc(f)
print(np.concatenate([A, G]).shape)
quit()
