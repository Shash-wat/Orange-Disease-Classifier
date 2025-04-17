import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import torch
import timm
import torchvision.transforms as transforms


model1 = tf.keras.models.load_model("Models/simple_cnn_model.keras")  
#model2 = tf.keras.models.load_model("Models/3_model_vgg.h5")
model3 = tf.keras.models.load_model("Models/model_convnext.h5")  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pt_convnext = timm.create_model('convnext_tiny', pretrained=True)
pt_convnext.reset_classifier(0)  
pt_convnext.eval().to(device)

pt_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_pt_features(image_path):
    """
    Uses timm's convnext_tiny (PyTorch) to extract a 768-dimensional feature vector from the image.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = pt_preprocess(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        features = pt_convnext(img_tensor) 
    return features.cpu().numpy()

classes = ["Citrus Canker", "Healthy", "Melanose"]

def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Prepare image for model1: Simple CNN expects 128x128 input
    img1 = preprocess_image(file_path, (128, 128))
    
    # Extract features for ConvNeXt classifier (model3) using PyTorch timm model.
    convnext_features = extract_pt_features(file_path)  # shape: (1, 768)
    
    # Get predictions
    predictions1 = model1.predict(img1)[0]
    # predictions2 = model2.predict(img2)[0]
    predictions3 = model3.predict(convnext_features)[0]
    
    # Format prediction results
    result_text1.set("Simple CNN Model Predictions:\n" + 
                     "\n".join([f"{classes[i]}: {predictions1[i] * 100:.2f}%" for i in range(len(classes))]))
    
    # result_text2.set("VGG Model Predictions:\n" + 
    #                  "\n".join([f"{classes[i]}: {predictions2[i] * 100:.2f}%" for i in range(len(classes))]))
    
    result_text3.set("ConvNeXt Model Predictions:\n" + 
                     "\n".join([f"{classes[i]}: {predictions3[i] * 100:.2f}%" for i in range(len(classes))]))
    
    # Display the uploaded image (resized for display)
    img_display = Image.open(file_path).convert("RGB")
    img_display = img_display.resize((128, 128))
    img_display = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_display)
    image_label.image = img_display

# ---------------------- Tkinter UI Setup ----------------------
root = tk.Tk()
root.title("Image Classifier - Compare 3 Models")
root.geometry("520x750")

title_label = Label(root, text="Upload an Image for Classification", font=("Arial", 14))
title_label.pack(pady=10)

upload_btn = Button(root, text="Upload Image", command=classify_image)
upload_btn.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_text1 = tk.StringVar()
result_label1 = Label(root, textvariable=result_text1, font=("Arial", 12), fg="blue")
result_label1.pack(pady=10)

# result_text2 = tk.StringVar()
# result_label2 = Label(root, textvariable=result_text2, font=("Arial", 12), fg="red")
# result_label2.pack(pady=10)

result_text3 = tk.StringVar()
result_label3 = Label(root, textvariable=result_text3, font=("Arial", 12), fg="green")
result_label3.pack(pady=10)

root.mainloop()

# VGG model is not uploaded , Train it to use for testing.