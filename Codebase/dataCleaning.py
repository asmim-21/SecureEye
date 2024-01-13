# SecureEye

# Data Cleaning
'''
The original data is in the form of two separate csv files, one for the purposes of training a dataset and the other for testing. 
In order to create a more robust machine learning model, I have chosen to combine these files into one single csv and then perform data cleaning on the whole dataset. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def combineData():
    # Reading CSV files into dataframes
    fraudTrain = pd.read_csv('Dataset/fraudTrain.csv')
    print("Trainging Dataset Size:", len(fraudTrain))
    fraudTest = pd.read_csv('Dataset/fraudTest.csv')
    print("Test Dataset Size:", len(fraudTest))

    # Combining datasets into one to create final dataset 
    combined_data = pd.concat([fraudTrain, fraudTest], ignore_index=True)
    combined_data.to_csv("Dataset/Fraud_Detection_Dataset.csv")

def readingDataset():
    dataset = pd.read_csv('Dataset/Fraud_Detection_Dataset.csv')
    return dataset

# Data cleaning
def dataClean(dataset):
    dataset.info()
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset.columns[0:2],axis=1) #not relevant
    dataset.info()

dataset = readingDataset()
dataClean(dataset)