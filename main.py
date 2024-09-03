import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Nustatome kelią iki aplankalų
train_data_path = r"C:\baigiamasis\Final_Training"
test_data_path = r"C:\baigiamasis\Final_Test"
test_csv_path = r"C:\baigiamasis\GT-final_test.csv"

# Funkcija treniravimo duomenų nuskaitymui
def load_training_data(base_path):
    data = []
    labels = []
    
    # Gauname visų aplankalų sąrašą
    folders = os.listdir(base_path)
    
    for folder in folders:
        # Pilnas aplankalo kelias
        folder_path = os.path.join(base_path, folder)
        
        # Patikriname, ar tai iš tikrųjų yra aplankalas
        if not os.path.isdir(folder_path):
            continue
        
        # Nuskaitymo žymių CSV failo kelias
        csv_file = os.path.join(folder_path, f"GT-{folder}.csv")
        csv_data = pd.read_csv(csv_file, sep=';')
        
        # Naudojame tik 'Filename' ir 'ClassId' stulpelius
        for index, row in csv_data[['Filename', 'ClassId']].iterrows():
            # Nuotraukos pilnas kelias
            img_path = os.path.join(folder_path, row['Filename'])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))  # Keičiame dydį į 32x32
            
            data.append(img)
            labels.append(row['ClassId'])
    
    return np.array(data), np.array(labels)

# Funkcija testavimo duomenų nuskaitymui
def load_test_data(data_path, csv_file):
    data = []
    labels = []
    csv_data = pd.read_csv(csv_file, sep=';')
    
    # Naudojame tik 'Filename' ir 'ClassId' stulpelius
    for index, row in csv_data[['Filename', 'ClassId']].iterrows():
        # Formuojame pilną nuotraukos kelią
        img_path = os.path.join(data_path, row['Filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))  # Keičiame dydį į 32x32
        data.append(img)
        labels.append(row['ClassId'])
        
    return np.array(data), np.array(labels)

# Nuskaitymas treniravimui ir testavimui
X_train, y_train = load_training_data(train_data_path)
X_test, y_test = load_test_data(test_data_path, test_csv_path)

# Normalizuojame duomenis
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Paverčiame žymes į kategorines
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

# Konvoliuciniai sluoksniai
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Pilnai susieti sluoksniai
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))  # Atnaujinta: Pakeistas neuronų skaičius į 43, kad atitiktų klasių skaičių

# Modelio kompiliavimas
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modelio santrauka
model.summary()

# Padaliname treniravimo duomenis į treniravimo ir validacijos rinkinius
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Modelio treniravimas
history = model.fit(X_train_split, y_train_split, epochs=30, batch_size=64, validation_data=(X_val_split, y_val_split))

# Modelio vertinimas
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Testavimo rinkinio tikslumas: {test_acc * 100:.2f}%")

# Vizualizuojame treniravimo istoriją
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

plt.show()
