from data_loader import load_training_data, load_test_data, train_data_path, test_csv_path, test_data_path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model, save_model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import os 

def start_training():
    def run_training(epochs):
        global model, history, X_train_split, y_train_split, X_val_split, y_val_split, X_test, y_test, datagen, test_acc
        try:
            X_train, y_train = load_training_data(train_data_path)
            X_test, y_test = load_test_data(test_data_path, test_csv_path)

            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.15,
                fill_mode="nearest"
            )

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            model = create_model()

            history = model.fit(
                datagen.flow(X_train_split, y_train_split, batch_size=64),
                validation_data=(X_val_split, y_val_split),
                epochs=epochs
            )

            test_loss, test_acc = model.evaluate(X_test, y_test)
            messagebox.showinfo("Success", f"Model trained! Test accuracy: {test_acc:.2f}")

            save_model(model, 'traffic_sign_model.h5')

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.title('Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='training_loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.title('Loss')

            plt.tight_layout()
            plt.show()

            y_test_pred = model.predict(X_test)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            y_test_true_classes = np.argmax(y_test, axis=1)

            report = classification_report(y_test_true_classes, y_test_pred_classes, target_names=[classes[i] for i in range(len(classes))])
            print("Classification Report:\n", report)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_epoch_selection():
        epoch_options = {
            '10': 10,
            '20': 20,
            '30': 30
        }

        choice = simpledialog.askstring("Select Epochs", "Enter the number of epochs:\n10 (5min)\n20 (7min)\n30 (10min)")

        if choice in epoch_options:
            epochs = epoch_options[choice]
            estimated_time = "5 minutes" if epochs == 10 else "7 minutes" if epochs == 20 else "10 minutes"
            messagebox.showinfo("Info", f"Training with {epochs} epochs. Estimated time: {estimated_time}.")
            threading.Thread(target=run_training, args=(epochs,)).start()
        else:
            messagebox.showwarning("Invalid Selection", "Please enter a valid number of epochs (10, 20, or 30).")

    get_epoch_selection()

def load_existing_model():
    global model
    try:
        model = load_model('traffic_sign_model.h5')
        messagebox.showinfo("Success", "Model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def upload_and_classify(result_label, image_label):
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

def show_data_distribution():
    try:
        folders = os.listdir(train_data_path)
        train_number = []
        class_num = []

        for folder in folders:
            train_files = os.listdir(os.path.join(train_data_path, folder))
            train_number.append(len(train_files))  
            class_num.append(classes[int(folder)])  

        zipped_lists = zip(train_number, class_num)
        sorted_pairs = sorted(zipped_lists) 

        train_number, class_num = zip(*sorted_pairs) 

        plt.figure(figsize=(21, 10))
        plt.bar(class_num, train_number)
        plt.xticks(rotation='vertical')
        plt.xlabel('Traffic Sign Classes')
        plt.ylabel('Number of Images')
        plt.title('Training Data Distribution (Sorted)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while generating the graph: {e}")


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