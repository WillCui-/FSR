# wyc2112

'''
Converts .mp4 files into 5 separate .wav stem files.

Runs the ffmpeg command on .mp4 files to split each channel into its own .wav stem file
'''

import os
import subprocess

from tqdm import tqdm

data_dir = './data'
stages = ['test', 'train']

for stage in stages:
    for i, file in tqdm(enumerate(sorted(os.listdir("{}/{}".format(data_dir, stage))))):
        for track in range(5):
            command = ["ffmpeg", "-i", "{}/{}/{}".format(data_dir, stage, file), "-vn", "-acodec", "pcm_s16le", "-map", "0:a:{}".format(
                track), "-ar", "44100", "{}/{}/{}_{}.wav".format(data_dir, stage+"_mod", i, track)]
            subprocess.run(command)
