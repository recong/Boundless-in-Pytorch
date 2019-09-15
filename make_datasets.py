import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--root", default='./', help="root pass")
parser.add_argument("--dataset_name", type=str, default="Pic", help="name of the dataset")
args = parser.parse_args()
os.makedirs(args.dataset_name, exist_ok=True)

with open('download_list.txt') as f:
    w = f.read()
    w = w.split('\n')
    n = 0
    print(w)
    for i in w:
        if os.path.exists(args.root + 'data_256/' + i[0] + '/' + i):
            img_list = sorted(list(glob.glob(args.root + 'data_256/' + i[0] + '/' + i + "/*.*")))
            for j in range(10):
                shutil.copy(img_list[j], "./Pic/" + str(n) + '.png')
                n = n + 1
        else:
            print(args.root + i[0] + '/' + i + 'does not exist')
