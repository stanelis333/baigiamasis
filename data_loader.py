import os
import cv2
import pandas as pd
import numpy as np

train_data_path = r"C:\baigiamasis\Final_Training"
test_data_path = r"C:\baigiamasis\Final_Test"
test_csv_path = r"C:\baigiamasis\Final_Test\GT-final_test.csv"

def load_training_data(base_path):
    data = []
    labels = []

    folders = os.listdir(base_path)
    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        csv_file = os.path.join(folder_path, f"GT-{folder}.csv")
        csv_data = pd.read_csv(csv_file, sep=';')

        for index, row in csv_data[['Filename', 'ClassId']].iterrows():
            img_path = os.path.join(folder_path, row['Filename'])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            data.append(img)
            labels.append(row['ClassId'])

    return np.array(data), np.array(labels)

def load_test_data(data_path, csv_file):
    data = []
    labels = []
    csv_data = pd.read_csv(csv_file, sep=';')

    for index, row in csv_data[['Filename', 'ClassId']].iterrows():
        img_path = os.path.join(data_path, row['Filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        data.append(img)
        labels.append(row['ClassId'])

    return np.array(data), np.array(labels)
