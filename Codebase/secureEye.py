# SecureEye

# Data Cleaning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readingDataset():
    dataset = pd.read_csv('../Dataset/fraudTrain.csv')
    return dataset

# Data cleaning
def dataClean(dataset):
    list(dataset)
    dataset.info()
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset.columns[0:1],axis=1) #not relevant
    print("\nCleaned Data\n")
    dataset.info()
        
# Exploratory Data Analysis
def EDA(dataset):
    print("\nSummary Statistics:")
    print(dataset.describe())


dataset = readingDataset()
dataClean(dataset)
EDA(dataset)