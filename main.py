import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from Dataloader import ImageDataset
import pickle
from Draw import Drawfunc
import dlib
import face_recognition
from tqdm import tqdm

'''
main function of Face comparing
developed by Cong Zou, 11/15/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py, Dataloader.py and Draw.py
CAUTION: These three .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/joy1.bmp
to find the image.
'''


parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type = str, default = 'face_dataset\\test')
parser.add_argument('--sample_dir', type = str, default = 'face_dataset\\sample')
parser.add_argument('--config_dir', type = str, default = 'config')
parser.add_argument('--data_mode', type = str, choices = ['jpg','p'], default = 'p')
parser.add_argument('--matrix_dir', type = str, default = 'config\\matrix.p')
parser.add_argument('--res_dir', type = str, default = None)

def main(args):
    if args.data_mode == 'jpg':
        args.test_dir = os.path.expanduser(args.test_dir)
        test = ImageDataset(args.test_dir)
        args.sample_dir = os.path.expanduser(args.sample_dir)
        sample = ImageDataset(args.sample_dir)
        print(len(test.datapaths), len(sample.datapaths))

        #calculate and save image encodings and distances
        args.config_dir = os.path.expanduser(args.config_dir)
        if os.path.exists(args.config_dir) == False:
            os.makedirs(args.config_dir)
        test_encoding = []
        sample_encoding = []
        pbar = tqdm(total = len(test.datapaths), desc = 'Image Encoding...')
        for i in range(len(test.datapaths)):
            temp = face_recognition.load_image_file(test.datapaths[i])
            test_encoding.append(face_recognition.face_encodings(temp)[0])
            temp = face_recognition.load_image_file(sample.datapaths[i])
            sample_encoding.append(face_recognition.face_encodings(temp)[0])
            pbar.update()
        pbar.close()
        #save the encoding information
        testpath = args.config_dir + '\\test.p'
        with open(testpath, 'wb') as file:
            pickle.dump(test_encoding, file)
        samplepath = args.config_dir + '\\sample.p'
        with open(samplepath, 'wb') as file:
            pickle.dump(sample_encoding, file)
        '''
        test_path = 'config\\test.p'
        sample_path = 'config\\sample.p'
        with open(test_path, 'rb') as file:
            test_encoding = pickle.load(file)
        with open(sample_path, 'rb') as file:
            sample_encoding = pickle.load(file)
        '''
        #calculate and save score matrix
        matrix = np.zeros([len(test_encoding),len(sample_encoding)])
        pbar = tqdm(total = len(test.datapaths), desc = 'Comparing Images...')
        for i in range(len(test_encoding)):
            matrix[i] = face_recognition.face_distance(sample_encoding, test_encoding[i])
            pbar.update()
        pbar.close()
        print(matrix)
        matrixpath = args.config_dir + '\\matrix.p'
        with open(matrixpath, 'wb') as file:
            pickle.dump(matrix, file)
        
        
    if args.data_mode == 'p':
        mpath = os.path.expanduser(args.matrix_dir)
        with open(mpath, mode = 'rb') as file:
            matrix = pickle.load(file)

        
    
    df = Drawfunc(matrix, args.res_dir)
    '''
    df.Draw_ROC()
    df.Draw_Distribution()
    '''
    df.Draw_CMC()
    
    
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


