from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os

from tkinter import *
from tkinter import filedialog, messagebox
import threading
import cv2
from PIL import Image, ImageTk

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

test_csv_path = r"C:\baigiamasis\Final_Test\GT-final_test.csv"
train_data_path = r"C:\baigiamasis\Final_Training"
test_data_path = r"C:\baigiamasis\Final_Test"

# Classes dictionary
classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

# Function to load training data
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

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))  

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

root = Tk()
root.title("CNN Training and Classification")
root.geometry("600x600")

def start_training():
    def run_training():
        global model, history, X_train_split, y_train_split, X_val_split, y_val_split, X_test, y_test, datagen, test_acc
        try:
            X_train, y_train = load_training_data(train_data_path)
            X_test, y_test = load_test_data(test_data_path, test_csv_path)

            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            datagen = ImageDataGenerator(
            )

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            model = create_model()

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                datagen.flow(X_train_split, y_train_split, batch_size=64),
                validation_data=(X_val_split, y_val_split),
                epochs=10,
                callbacks=[early_stopping]
            )
            test_loss, test_acc = model.evaluate(X_test, y_test)
            accuracy_label.config(text=f"Test Accuracy: {test_acc * 100:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    threading.Thread(target=run_training).start()

def upload_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (32, 32))  
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_class_name = classes[predicted_class]
        
        img_display = Image.open(file_path)
        img_display = img_display.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        result_label.config(text=f"Predicted Class: {predicted_class_name}")

train_button = Button(root, text="Start Training", command=start_training)
train_button.pack(pady=20)

upload_button = Button(root, text="Upload and Classify Image", command=upload_and_classify)
upload_button.pack(pady=20)

accuracy_label = Label(root, text="")
accuracy_label.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
