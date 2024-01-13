# SecureEye

# Data Cleaning
'''
The original data is in the form of two separate csv files, one for the purposes of training a dataset and the other for testing. 
In order to create a more robust machine learning model, I have chosen to combine these files into one single csv and then perform data cleaning on the whole dataset. 
'''

import pandas as pd

# Read CSV files into dataframes
fraudTrain = pd.read_csv('Dataset/fraudTrain.csv')
print("Trainging Dataset Size:", fraudTrain.size)
fraudTest = pd.read_csv('Dataset/fraudTest.csv')
print("Test Dataset Size:", fraudTest.size)

# Combine these two datasets into one and create a new output file
combined_data = pd.concat([fraudTrain, fraudTest], ignore_index=True)
print("Final Dataset Size:", combined_data.size)
combined_data.to_csv("Dataset/Fraud_Detection_Dataset.csv")