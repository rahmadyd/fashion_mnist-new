import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Pastikan direktori untuk menyimpan model ada
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/fashion_mnist_model.pth"

# Tentukan perangkat (CPU atau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisasi untuk grayscale
])

# Dataset
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Pastikan dataset berhasil dimuat
assert len(train_data) > 0, "Dataset pelatihan kosong!"
assert len(test_data) > 0, "Dataset pengujian kosong!"

# DataLoader
train_Dloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_Dloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Definisi model
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
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
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

# Inisialisasi model
convModel = FashionConvNet().to(device)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(convModel.parameters(), lr=1e-3)

# Fungsi pelatihan
def training_convnet(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0

    for index, (image, label) in enumerate(dataloader):
        # Periksa apakah batch kosong
        if len(image) == 0:
            continue

        image, label = image.to(device), label.to(device)  # Pindahkan data ke perangkat
        pred = model(image)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Hitung total loss dan akurasi
        total_loss += loss.item()
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        # Logging setiap 100 batch
        if index % 100 == 0:
            loss_value = loss.item()
            current = (index + 1) * len(image)  # Jumlah data yang telah diproses sejauh ini
            if current > 0:  # Pastikan tidak membagi dengan nol
                accuracy = (correct / current) * 100
                print(f"Batch {index}: Loss: {loss_value:.3f}, Accuracy: {accuracy:.2f}%")
            else:
                print(f"Batch {index}: Loss: {loss_value:.3f}, Accuracy: N/A (current=0)")

    # Hitung rata-rata loss dan akurasi untuk epoch ini
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = (correct / size) * 100
    print(f"Training Loss: {avg_loss:.3f}, Training Accuracy: {avg_accuracy:.2f}%")
    return avg_loss, avg_accuracy

# Fungsi pengujian
def testing_convnet(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)  # Pindahkan data ke perangkat
            pred = model(image)
            total_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / size) * 100
    print(f"Test Loss: {avg_loss:.3f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# Loop pelatihan dan pengujian
epochs = 7
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for e in range(epochs):
    print(f"\n\nEpoch {e+1}/{epochs}\n")

    # Training
    train_loss, train_accuracy = training_convnet(train_Dloader, convModel, loss_fn, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Testing
    test_loss, test_accuracy = testing_convnet(test_Dloader, convModel, loss_fn)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Simpan model
torch.save(convModel.state_dict(), model_path)
print(f"Model telah disimpan di {model_path}")

# Plot Loss dan Akurasi
plt.figure(figsize=(12, 5))

# Plot Training and Test Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

# Plot Training and Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()