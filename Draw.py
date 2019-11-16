from matplotlib import pyplot as plt
import numpy as np
import pylab as pl
from scipy import interpolate
from tqdm import tqdm
import os
import pickle
from scipy.spatial.distance import hamming
import copy

class Drawfunc(object):
    def __init__(self, matrix, res_dir):
        super(Drawfunc, self).__init__()
        self.score = matrix
        self.res = res_dir
        if self.res != None:
            if os.path.exists(self.res) == False:
                os.makedirs(self.res)

    def Draw_ROC(self):
        true = []
        false = []
        #build TMR and FMR list
        for i in range(len(self.score)):
            for j in range(len(self.score[0])):
                if i == j:
                    true.append(self.score[i][j])
                else:
                    false.append(self.score[i][j])
        true_len = len(true)
        false_len = len(false)

        tmr = []
        fmr = []
        for i in range(101):
            true_c = 0
            false_c = 0
            standard = 1 - i/100
            for score in true:
                if score <= standard:
                    true_c += 1
            tmr.append(true_c/true_len)

            for score in false:
                if score <= standard:
                    false_c += 1
            fmr.append(false_c/false_len)
            
        print(true_len)
        print(false_len)

        plt.plot(fmr, tmr, '-b', label = 'ROC Curve')
        plt.title('ROC Curve')
        plt.xlabel('FMR')
        plt.ylabel('TMR')
        plt.xlim(-0.1, 1.1)
        plt.ylim(0.0, 1.1)
        plt.legend()
        if self.res != None:
            imgpath = self.res + '\\ROC_Curve.jpg'
            plt.savefig(imgpath)
        plt.show()
            
        return

    def Draw_Distribution(self):
        true = []
        false = []
        #build TMR and FMR list
        for i in range(len(self.score)):
            for j in range(len(self.score[0])):
                if i == j:
                    true.append(self.score[i][j])
                else:
                    false.append(self.score[i][j])
        true_len = len(true) #145
        false_len = len(false) #145*144
        print(true_len, false_len)
        
        
        true.sort()
        false.sort()

        ytrue = []
        yfalse = []

        tstart = 0
        fstart = 0
        
        for i in range(21):
            standard = i*0.05
            tempt = tstart
            tempf = fstart
            if tstart < true_len - 1:
                while true[tstart] <= standard:
                    tstart += 1
                    if tstart >= true_len:
                        break
            ytrue.append((tstart - tempt)/true_len)
            if fstart < false_len - 1:
                while false[fstart] <= standard:
                    fstart += 1
                    if fstart >= false_len:
                        break
            yfalse.append((fstart - tempf)/false_len)

        x = np.arange(0, 1.05, 0.05)
        
        plt.plot(x, ytrue, '-r', label = 'Genuine')
        plt.plot(x, yfalse, '-b', label = 'Imposter')


        plt.title('Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

        plt.xlim(-0.1, 1.1)
        plt.ylim(0.0, 1.0)

        plt.legend()
        if self.res != None:
            imgpath = self.res + '\\Distribution.jpg'
            plt.savefig(imgpath)
        plt.show()
        return

    def Draw_CMC(self):
        #first sort matrix score in the order of descending
        #then check whether the lowest score has been less than the score of true
        #find how many false cases should be taken to get the true cases
        matrix = copy.deepcopy(self.score)
        for i in range(len(matrix)):
            matrix[i].sort()
        x = np.arange(0,len(matrix) + 1)
        y = [0]
        for j in range(len(matrix[0])):
            count = 0
            for i in range(len(matrix)):
                if matrix[i][j] >= self.score[i][i]:
                    count += 1
            y.append(count/len(matrix))

        plt.plot(x, y, '-b', label = 'CMC Curve')
        plt.title('CMC Curve')
        plt.xlabel('RANK')
        plt.ylabel('CMC')
        plt.xlim(-1, len(matrix) + 1)
        plt.ylim(0.0, 1.1)
        plt.legend()
        
        if self.res != None:
            imgpath = self.res + '\\CMC_Curve.jpg'
            plt.savefig(imgpath)
        plt.show()
                
