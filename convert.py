import os
import tqdm

data_dir = './data'
stages = ['test', 'train']

for stage in stages:
    for i, file in tqdm(enumerate(os.listdir("{}/{}".format(data_dir, stage)))):
        os.system(
            "ffmpeg -i test/{} -vn -acodec pcm_s16le -map 0:a:1 -ar 44100 {}/{}.wav".format(file, stage+"_mod", i))
