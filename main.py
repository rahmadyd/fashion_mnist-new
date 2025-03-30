import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Daftar label untuk Fashion MNIST
LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Model class
class FashionConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 6, 512)  # Perbarui dimensi input
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(0.2)
        self.batchNorm = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.batchNorm(x)
        x = self.out(x)
        return x

# Load pretrained model
MODEL_PATH = "C:/Users/rahma/AI_Study/fashion_mnist-new/saved_models/fashion_mnist_model.pth"
model = FashionConvNet()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Tambahkan batch dimension
    return image

# Streamlit UI
st.title("Fashion MNIST Image Classifier")
st.write("Upload an image of a fashion item, and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("\n")

        if st.button("Predict"):  # Tombol untuk memulai prediksi
            image_tensor = preprocess_image(image)
            image_tensor = image_tensor.to(torch.device('cpu'))  # Pastikan tensor berada di CPU
            with torch.no_grad():
                prediction = model(image_tensor)
                predicted_label = torch.argmax(prediction, dim=1).item()
                st.write(f"Prediction: {LABELS[predicted_label]}")
    except Exception as e:
        st.error(f"Error processing image: {e}")