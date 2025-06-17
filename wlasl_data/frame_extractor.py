import json
import subprocess
import os
import random
#import ffmpeg

home = "C:/Users/atrai/OneDrive - Lehigh University/Desktop/Summer 25/Code/ALSR-CNN/wlasl_data/"
vid_path = "videos/"
frame_path = "extracted/"
frame_rate = 5
# command -> ffmpeg -i <src> -vf fps=<frames>/<seconds> <dest>

def extract_frames(src, dest, testing=False, test_range=5, rand_set=False):
    files = os.listdir(src)
    if(rand_set):
        files = random.sample(files, test_range)
    
    if(testing):
        for i in range(test_range):
            file = files[i]
            print(file)
            if(not os.path.isdir(f"{dest}/{file[:5]}")):
                os.mkdir(f"{dest}/{file[:5]}")
            out_path = f"{dest}/{file[:5]}/frame_%03d.jpg"
            cmd = ['ffmpeg', '-i', f'{src}/{file}', '-vf', f'fps={frame_rate}/1', f'{out_path}']
            subprocess.run(cmd)
    else:
        for file in files:
            out_path = f"{dest}/{file[:5]}/frame_%03d.jpg"
            cmd = ['ffmpeg', '-i', f'{src}/{file}', '-vf', f'fps={frame_rate}/1', f'{out_path}']
    print('[DEBUG:] Done')
    
    
extract_frames("./videos", "./extracted", testing=True, test_range=100, rand_set=True)
