from tkinter import *
from gui_functions import start_training, load_existing_model, upload_and_classify, show_data_distribution

root = Tk()
root.title("CNN Training and Classification")
root.geometry("600x600")

accuracy_label = Label(root, text="")
accuracy_label.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="")
result_label.pack(pady=10)

train_button = Button(root, text="Train New Model", command=start_training)
train_button.pack(pady=20)

load_button = Button(root, text="Load Existing Model", command=load_existing_model)
load_button.pack(pady=20)

upload_button = Button(root, text="Upload and Classify Image", command=lambda: upload_and_classify(result_label, image_label))
upload_button.pack(pady=20)

distribution_button = Button(root, text="Show Data Distribution", command=show_data_distribution)
distribution_button.pack(pady=20)

root.mainloop()
