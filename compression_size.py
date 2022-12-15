# Add LPC computation here

print("Loading...")
f = './data/test_mod/{}.wav'.format('0_0')
[_, pcm_data] = scipy.io.wavfile.read(f)
print(pcm_data.shape)

A, G = wav_to_lpc(f)
print(np.concatenate([A, G]).shape)
quit()
