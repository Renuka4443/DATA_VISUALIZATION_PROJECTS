#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


# In[2]:


#set up directories
train_dir = r"C:/Users/dakho/Downloads/archive (4)/valid"
valid_dir = r"C:/Users/dakho/Downloads/archive (4)/valid" 
test_dir=r"C:/Users/dakho/Downloads/archive (4)/test"

train_datagen=ImageDataGenerator(rescale=1./255)
valid_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(64,64),batch_size=32,class_mode='binary')
valid_generator=valid_datagen.flow_from_directory(valid_dir,target_size=(64,64),batch_size=32,class_mode='binary')
test_generator=test_datagen.flow_from_directory(test_dir,target_size=(64,64),batch_size=32,class_mode='binary')


# In[3]:


#building a simple CNN model 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') #binary classification: wildfire or no wildfire
    ])

#compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[4]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True


# In[5]:


history=model.fit(train_generator,validation_data=valid_generator,epochs=5,verbose=1)


# In[6]:


model.save("ffd_model.h5")


# In[7]:


from tensorflow.keras.models import load_model


# In[8]:


model=load_model("ffd_model.h5")
print("model loaded successfully!")


# In[9]:


# Function to load and predict an image
def predict_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the image in the GUI
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)    #convert image for tk
        image_label.configure(image=img) #update the image in GUI
        image_label.image = img

        # Preprocess the image for the model
        img_for_model = Image.open(file_path).resize((64, 64))
        img_array = np.array(img_for_model) / 255.0  # Rescale like during training
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)[0][0] #extracts the scalar prediction value
        result = "Wildfire" if prediction > 0.5 else "No Wildfire"
        result_label.config(text="Prediction: " + result)

# Setting up the GUI window
root = tk.Tk()
root.title("Forest Fire Detection")
root.geometry("400x400")

# Add widgets
btn = tk.Button(root, text="Upload Image", command=predict_image) #button triggers the predict_image() function when clicked
btn.pack(pady=20)

#Placeholder for displaying the selected image
image_label = tk.Label(root)
image_label.pack()

#Label to display the prediction result
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=20)

#Starts the Tkinter event loop, keeping the GUI active until manually closed
root.mainloop()

