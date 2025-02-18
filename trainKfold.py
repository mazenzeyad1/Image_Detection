from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
import os
import shutil
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_DIR = r"Dataset2"
#test_DIR = r"Dataset\Test"

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batch_size = 32
epochs = 10
learning_rate = 0.001
num_classes = 6
#print(device)
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust size as needed for your model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet standard
])


# Define the new root directory for the flattened structure
flattened_root = "flattened_dataset"

# Create the new root directory if it doesn't exist
if not os.path.exists(flattened_root):
    os.makedirs(flattened_root)

import shutil

# Iterate through each class directory in the original nested structure
for class_dir in os.listdir(train_DIR):
    class_path = os.path.join(train_DIR, class_dir)

    # Iterate through each subdirectory in the class directory
    for subdir1 in os.listdir(class_path):
        subdir1_path = os.path.join(class_path, subdir1)

        # Iterate through each image file in the sub-subdirectory
        for img_file in os.listdir(subdir1_path):
            img_path = os.path.join(subdir1_path, img_file)

            # Define the new directory structure and filename for the image
            new_dir = os.path.join(flattened_root, f"{class_dir}_{subdir1}")
            new_filename = f"{class_dir}_{subdir1}_{img_file}"

            # Create the new directory if it doesn't exist
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Copy the image file to the new directory with the new filename
            shutil.copy(img_path, os.path.join(new_dir, new_filename))
# Now use datasets.ImageFolder with the flattened directory structure
train_dataset = datasets.ImageFolder(root=flattened_root, transform=transform)



# #Loading the dataset:
# train_dataset = datasets.ImageFolder(root=train_DIR, transform=transform)

# =============================================================================


        
def train(model, train_loader, test_loader, epochs, learning_rate, device, fold_number):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_accuracy_model = test(model, test_loader, device)
        if test_accuracy_model > best_accuracy:
            best_accuracy = test_accuracy_model
            torch.save(model.state_dict(), f'weights/best_{fold_number}.pt')
            
        print(f"current Best Val. Acc: {best_accuracy}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
# =============================================================================
def test(model, test_loader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(correct)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
# =============================================================================

def calculate_overall_precision(model, test_loader, device):
  """
  Calculates overall precision for a multi-class classification task.

  Args:
      model: The PyTorch model to evaluate.
      test_loader: The data loader for the test set.
      device: The device to use for computations ("cpu" or "cuda").

  Returns:
      The overall precision as a float.
  """

  model.eval()
  true_positives = 0
  total_predictions = 0

  with torch.no_grad():
    for data in test_loader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      # Count true positives
      for i, (label, prediction) in enumerate(zip(labels, predicted)):
        if label == prediction:
          true_positives += 1

      total_predictions += labels.size(0)

  # Calculate overall precision
  if total_predictions > 0:  # Avoid division by zero
    precision = true_positives / total_predictions
  else:
    precision = 0.0

  return 100 *precision
# =============================================================================
def calculate_overall_recall(model, test_loader, device):
  """
  Calculates overall recall for a multi-class classification task.

  Args:
      model: The PyTorch model to evaluate.
      test_loader: The data loader for the test set.
      device: The device to use for computations ("cpu" or "cuda").

  Returns:
      The overall recall as a float.
  """

  model.eval()
  true_positives = 0
  total_positives = 0

  with torch.no_grad():
    for data in test_loader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      # Count true positives and total positives per class
      for i, (label, prediction) in enumerate(zip(labels, predicted)):
        if label == 1:  # Assuming positive class label is 1
          total_positives += 1
          if label == prediction:
            true_positives += 1

  # Calculate overall recall
  if total_positives > 0:  # Avoid division by zero
    recall = true_positives / total_positives
  else:
    recall = 0.0

  return 100 * recall
# =============================================================================
def calculate_f1_score(model, test_loader, device):
  """
  Calculates overall F1 score for a multi-class classification task.

  Args:
      model: The PyTorch model to evaluate.
      test_loader: The data loader for the test set.
      device: The device to use for computations ("cpu" or "cuda").

  Returns:
      The overall F1 score as a float.
  """

  model.eval()
  true_positives = 0
  total_predictions = 0
  total_positives = 0

  with torch.no_grad():
    for data in test_loader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      # Count true positives, predictions, and total positives per class
      for i, (label, prediction) in enumerate(zip(labels, predicted)):
        if label == 1:  # Assuming positive class label is 1
          total_positives += 1
          if label == prediction:
            true_positives += 1
        total_predictions += 1

  # Calculate precision and recall (handle division by zero)
  precision = 0.0 if total_predictions == 0 else true_positives / total_predictions
  recall = 0.0 if total_positives == 0 else true_positives / total_positives

  # Calculate F1 score (harmonic mean of precision and recall)
  if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
  else:
    f1_score = 0.0

  return 100 *f1_score

# =============================================================================
def main():
    # Assuming class names are available in the dataset
    # Define the class names based on the directory structure
    class_names = train_dataset.classes
    print(class_names)

    print("model K-Fold Cross-Validation:-")
    torch.manual_seed(42)
    kfold = KFold(n_splits=10, shuffle=True)  # Adjust k as needed
    
    # No need for separate train and test loaders here
    total_val_acc = 0
    total_val_perc = 0
    total_val_rec = 0
    total_val_f1 = 0
    tot_f1 = 0

    for fold_number, (train_index, val_index) in enumerate(kfold.split(train_dataset), 1):
        print(f"Fold number: {fold_number}")
        # Create data loaders for each fold based on train_index and val_index
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)

        # Create a new model model instance for each fold (improves memory usage)
        model = New(num_classes).to(device)

        # Train the model on the current fold
        train(model, train_loader, val_loader, epochs, learning_rate, device, fold_number)
        
        # Load best weights for evaluation
        weights_path = f"weights/best_{fold_number}.pt"
        # Create a new VGG16 model (ensure it matches the architecture of your saved weights)
        model =  New(num_classes).to(device)

        
        # Load the saved weights into the model
        model_state_dict = torch.load(weights_path)
        model.load_state_dict(model_state_dict)

        # Calculate fold validation accuracy and accumulate total
        fold_val_acc = test(model, val_loader, device)
        total_val_acc += fold_val_acc
        print(f"Fold Validation Accuracy: {fold_val_acc:.2f}%")

        # Calculate fold validation precision and accumulate total
        fold_val_perc = calculate_overall_precision(model, val_loader, device)
        total_val_perc += fold_val_perc
        print(f"Fold Validation Percision: {fold_val_perc:.2f}%")
        
        # Calculate fold validation recall and accumulate total
        fold_val_rec = calculate_overall_recall(model, val_loader, device)
        total_val_rec += fold_val_rec
        print(f"Fold Validation Recall: {fold_val_rec:.2f}%")

        # Calculate fold validation f1 score and accumulate total
        fold_val_f1 = calculate_f1_score(model, val_loader, device)
        total_val_f1 += fold_val_f1
        print(f"Fold Validation F1 score: {fold_val_f1:.2f}%")
        
             # Calculate confusion matrix
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - Fold {fold_number}")
        plt.show()
        
        print("==============================================================")
        
        
    # Print final average validation accuracy across all folds
    avg_val_acc = total_val_acc / len(kfold)
    print(f"\nAverage Validation Accuracy: {avg_val_acc:.2f}%")
    
    # Print final average validation percision across all folds
    avg_val_perc = total_val_perc / len(kfold)
    print(f"\nAverage Validation Accuracy: {avg_val_perc:.2f}%")
    
    # Print final average validation recall across all folds
    avg_val_rec = total_val_rec / len(kfold)
    print(f"\nAverage Validation Accuracy: {avg_val_rec:.2f}%")
    
    # Print final average validation f1 score across all folds
    avg_val_f1 = total_val_f1 / len(kfold)
    print(f"\nAverage Validation Accuracy: {avg_val_f1:.2f}%")



if __name__ == "__main__":
    main()                         