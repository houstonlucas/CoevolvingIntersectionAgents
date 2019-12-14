import gym
import gym_fluids
import os
from PIL import Image, ImageDraw

path = '/home/gaetano/Desktop/images/'
sub_directories = []
dir_name = []
for r, d, _ in os.walk(path):
    for direct in d:
        dir_name.append(direct)
        sub_path = os.path.join(r, direct)
        files = []
        for root, directory, file in os.walk(sub_path):
            for fi in file:
                files.append(os.path.join(root, fi))
        files.sort()
        sub_directories.append(files)

for i, directory in enumerate(sub_directories):
    images = []
    for file in directory:
        img = Image.open(file)
        crop_img = img.crop((0, 0, 800, 800))
        images.append(crop_img)
    name = path + dir_name[i] + '.gif'
    images[0].save(name, format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)



