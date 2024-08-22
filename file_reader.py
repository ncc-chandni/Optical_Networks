'''
Reading the CSV File: Both methods use Python's built-in csv module to 
            read data from a CSV file. The data is split into two parts: 
            features (samples) and labels (osnr_labels). 
            The features are all columns except the last one, and 
            the labels are the values in the last column.
Converting Data to Numpy Arrays: After processing, the features and 
        labels are converted into numpy arrays (X and y), which are 
        commonly used formats in machine learning for input data.
''' 

'''
After processing, the features and labels are converted into numpy arrays 
(X and y), which are commonly used formats in machine learning for input data
'''

import csv
import numpy as np 
import pandas as pd

class FileReader():
    def read_array_two_class(self, _file):
        df = pd.read_csv(_file)
        X = df.iloc[:,:-1].values
        osnr_labels = df.iloc[:, -1].values
        y = np.where(osnr_labels < 17, [1,-1], [-1,1])
        return X, y 
    
    def read_array_three_class(self, _file):
        df = pd.read_csv(_file)
        X = df.iloc[:, :-1].values 
        osnr_labels = df.iloc[:, -1].values
        y = np.zeros((len(osnr_labels), 4))

        y[osnr_labels >= 17, 0] = 1 
        y[(osnr_labels >= 14) & (osnr_labels < 17), 1] = 1
        y[(osnr_labels>=10) & (osnr_labels < 14), 2] = 1 
        y[osnr_labels < 10, 3] = 1 
        #classA = 1000, classB = 0100, classC = 0010, classD = 0001

        classA = np.sum(osnr_labels >= 17)
        classB = np.sum((osnr_labels >= 14) & (osnr_labels < 17))
        classC = np.sum((osnr_labels>=10) & (osnr_labels < 14))
        classD = np.sum(osnr_labels < 10)

        total_samples = classA + classB + classC + classD 
        print(f"Class A samples : {classA} \nClass B samples : {classB} \nClass C samples : {classC} \nClass D samples : {classD}")

        return X, y 
    


    