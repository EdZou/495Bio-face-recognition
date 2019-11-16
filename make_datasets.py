import os
import cv2
from tqdm import tqdm
import imghdr
import argparse
from Dataloader import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = '495_dataset')

def main(args):
    #make directories
    final_dataset = 'face_dataset'
    ID = ImageDataset(args.dataset)
    imgpaths = ID.datapaths
    final_dataset = os.path.expanduser(final_dataset)
    if os.path.exists(final_dataset) == False:
        os.makedirs(final_dataset)
    test_dir = final_dataset + '\\test'
    sample_dir = final_dataset + '\\sample'
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)
    if os.path.exists(sample_dir) == False:
        os.makedirs(sample_dir)
        
    #make a new dataset
    temp = ''
    pbar = tqdm(total = 150, desc = 'Making New Dataset...')
    for i in range(len(imgpaths)):
        record = ''
        for j in range(14):
            if imgpaths[i][-13+j] >= '0' and imgpaths[i][-13+j] <= '9':
                record += imgpaths[i][-13+j]
            if imgpaths[i][-13+j] == 'd':
                break
        if temp == '' or  record != temp:
            temp = record
            img = cv2.imread(os.path.expanduser(imgpaths[i]))
            tempath = test_dir + '\\' + temp + '.jpg'
            cv2.imwrite(tempath, img)
            img = cv2.imread(os.path.expanduser(imgpaths[i+1]))
            tempath = sample_dir + '\\' + temp + '.jpg'
            cv2.imwrite(tempath, img)
            pbar.update()
    pbar.close()
    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
