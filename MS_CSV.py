#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
#import random
import math
#from skimage import data
#from skimage.transform import resize
import argparse
import pandas as pd
#from scipy import stats



parser = argparse.ArgumentParser(description= 'Preprocess .ms files')
parser.add_argument('number', type=int, help= 'Number of .ms files of the chosen class')
parser.add_argument('class_type',type=int, help= '1 for generating sweep training files, 0 for generating neutral training files')
parser.add_argument('ts_tr',type=int, help= '1 for Testing samples training files, 0 for Training sample')



args = parser.parse_args()





num_ = args.number
class_label= args.class_type
Tr_Ts=args.ts_tr


sub = "sweep_" if class_label == 1 else "neut_"
div = "test" if Tr_Ts == 1 else "train"




def addIndividuals(np_array):

    indexes = range(np_array.shape[0])
    list_array_added = []

    for j in indexes:
        if ((j % 2) != 0):
            array_temp = np_array[j,:] + np_array[j-1,:]
            list_array_added.append(array_temp)

    

    np_array_added = np.array(list_array_added)

    return np_array_added



def saveToCSV(i):
    filename_ms = f"./Data/MS_files_{div}/{sub}{i+1}.ms"
    filename_csv = f"./Data/MS_files_{div}/{sub}{div}_{i+1}.csv"

    file = open(filename_ms, 'r')
    Lines = file.readlines()
    count = 0
    genetic = False

    list_info = []

    # Strips the newline character
    for line in Lines:
        if ('positions:' in line):
            genetic = True

        elif ('//' in line):
            genetic = False

        elif (genetic == True):
            info = list(line[:-1])
            if (info != []):
                info_np = np.array(info, dtype=int)
                list_info.append(info_np)

    np_array = np.array(list_info)

#    np_array_added = addIndividuals(np_array)

    np.savetxt(filename_csv, np_array, delimiter=",")



def main():
    maxIterations = num_


    for i in range(num_):
        print('Saving CSV Iteration: ', i)
        saveToCSV(i)


if __name__ == "__main__":
    main()


# In[ ]:




