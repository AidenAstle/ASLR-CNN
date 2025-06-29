import json
import subprocess
import os
import sys
import shutil
import random
#import ffmpeg

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
        if(len(os.listdir(dest)) > 1):   # if dest is not empty
            for file in os.listdir(dest):      # remove all folders and files in dest folder
                path = f"{dest}/{file}"
                print(f'Removing {path}')
                shutil.rmtree(path)
        for file in files:  # run for all video files
            out_path = f"{dest}/{file[:5]}/frame_%03d.jpg"
            cmd = ['ffmpeg', '-i', f'{src}/{file}', '-vf', f'fps={frame_rate}/1', f'{out_path}']
            if(not os.path.isdir(f"{dest}/{file[:5]}")):
                os.mkdir(f"{dest}/{file[:5]}")
            subprocess.run(cmd)
    print('[DEBUG:] Done')
    
if __name__ == '__main__':
    vid_path = sys.argv[1]
    dest_path = sys.argv[2]
    frame_rate = int(sys.argv[3])
    extract_frames(vid_path, dest_path, frame_rate)
    if sys.argv[1] == '--b':  # build all files
        extract_frames("./videos", "./extracted", testing=False)
# extract_frames("./videos", "./extracted", testing=True, test_range=100, rand_set=True)