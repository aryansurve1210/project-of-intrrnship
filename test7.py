import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tensorflow as tf
from PIL import Image, ImageTk

# Function to predict Instagram profile
def predict_instagram_profile(data):
    # Convert dictionary to DataFrame
    user_data = pd.DataFrame([data])

    # Scaling user data
    user_data_scaled = scaler.transform(user_data)

    # Predict user data
    user_pred = model.predict(user_data_scaled)
    user_pred_label = "Fake" if np.argmax(user_pred) == 1 else "Not Fake"
    messagebox.showinfo("Prediction", f"The provided Instagram profile is likely {user_pred_label}.")

# Function to switch to the prediction page
def show_prediction_page():
    welcome_frame.pack_forget()
    prediction_frame.pack()

# Create the main window
root = tk.Tk()
root.title("Welcome Page")
root.geometry("300x200")

# Welcome page frame
welcome_frame = tk.Frame(root)
welcome_frame.pack()

# Welcome message
welcome_label = tk.Label(welcome_frame, text="Welcome to the Application!")
welcome_label.pack(pady=20)

# Button to open the prediction page
prediction_button = tk.Button(welcome_frame, text="Predict Instagram Profile", command=show_prediction_page)
prediction_button.pack()

# Prediction page frame
prediction_frame = tk.Frame(root)

# Load the training dataset
instagram_df_train = pd.read_csv(r"C:\final_year_project - Copy\insta_train.csv")

# Preparing Data to Train the Model
# Training dataset (inputs)
X_train = instagram_df_train.drop(columns=['fake'])

# Training dataset (Outputs)
y_train = instagram_df_train['fake']

# Scaling the data before training the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

# Building and Training Deep Learning Model
model = Sequential([
    Dense(50, input_dim=11, activation='relu'),
    Dense(150, activation='relu'),
    Dropout(0.3),
    Dense(150, activation='relu'),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs_hist = model.fit(X_train_scaled, y_train, epochs=50, verbose=1, validation_split=0.1)

# GUI Setup for prediction page
prediction_frame = tk.Frame(root)

# Create input fields for each feature
entries = {}
for i, feature in enumerate(['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username',
                'description length', 'external URL', 'private', '#posts', '#followers', '#follows']):
    label = ttk.Label(prediction_frame, text=feature)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = ttk.Entry(prediction_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[feature] = entry

# Button to predict
predict_button = ttk.Button(prediction_frame, text="Predict", command=lambda: predict_instagram_profile({feature: float(entry.get()) for feature, entry in entries.items()}))
predict_button.grid(row=11, column=0, columnspan=2, pady=10)

root.mainloop()
