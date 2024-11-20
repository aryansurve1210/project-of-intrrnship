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
def predict_instagram_profile():
    # Create dictionary to store user input
    data = {}

    # Prompt user for input for each feature
    for feature in ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username',
                    'description length', 'external URL', 'private', '#posts', '#followers', '#follows']:
        data[feature] = float(entries[feature].get())

    # Convert dictionary to DataFrame
    user_data = pd.DataFrame([data])

    # Convert numeric columns to numeric, ignoring non-numeric values
    user_data = user_data.apply(pd.to_numeric, errors='ignore')

    # Scaling user data
    user_data_scaled = scaler.transform(user_data)

    # Predict user data
    user_pred = model.predict(user_data_scaled)
    user_pred_label = "Fake" if np.argmax(user_pred) == 1 else "Not Fake"
    messagebox.showinfo("Prediction", f"The provided Instagram profile is likely {user_pred_label}.")

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

# GUI Setup
root = tk.Tk()
root.title("Fake Instagram Profile Detection")
root.geometry("400x400")

# Load and display background image
background_image = Image.open(r'C:\final_year_project\images\4444444444444444.jpg')
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Create input fields for each feature
entries = {}
for i, feature in enumerate(['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username',
                'description length', 'external URL', 'private', '#posts', '#followers', '#follows']):
    label = ttk.Label(root, text=feature)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[feature] = entry

# Button to predict
predict_button = ttk.Button(root, text="Predict", command=predict_instagram_profile)
predict_button.grid(row=11, column=0, columnspan=2, pady=10)

root.mainloop()
