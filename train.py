import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vgg
from models import *

train_DIR = r"ds/train"
test_DIR = r"ds/test"

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batch_size = 32
epochs = 200
learning_rate = 0.001
num_classes = 2
#print(device)
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust size as needed for your model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet standard
])
# Loading the dataset:
train_dataset = datasets.ImageFolder(root=train_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=test_DIR, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
# =============================================================================
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)
            #print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()



        val_acc =test(model, test_loader, device)
        print(f"current  Val. Acc: {val_acc}")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}, Val. Acc: {val_acc}")
        # Save model with best accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best-vgg16.pt')
        print(f"current Best Val. Acc: {best_accuracy}")

# =============================================================================
def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
# =============================================================================


def main():
    print("VGG-16:-")
    torch.manual_seed(42)
    vgg16 = VGG16(num_classes).to(device)
    train(vgg16, train_loader, epochs, learning_rate, device)
    
    torch.save(vgg16.state_dict(), 'latest-vgg16.pt')
# =============================================================================
# =============================================================================  
    test_accuracy = test(vgg16, test_loader, device)
    print(f"model accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
