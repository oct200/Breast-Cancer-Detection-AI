import streamlit as st
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

@st.cache_resource
def incarcare_model():
    print("Loading model")

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)

    model.load_state_dict(torch.load('mamografii.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.to(DEVICE)
    print("model loaded")
    return model

def evaluareImagine(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        print("Predicted class:", predicted.item())

    class_map = {0: "no tumor", 1: "tumor"}
    print("Result:", class_map[predicted.item()])
    return class_map[predicted.item()]


def init_GUI():
    file_path = st.file_uploader("Choose an image")
    if file_path is not None:
        st.image(file_path,width=300)
        if st.button("Evaluare"):
            st.text(evaluareImagine(file_path))


def init_var():
    pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = incarcare_model()
init_var()
init_GUI()