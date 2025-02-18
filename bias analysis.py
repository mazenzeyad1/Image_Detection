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

test_DIR = r"Test"

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#batch_size = 32
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
flattened_root = "flattened_Test"

# Create the new root directory if it doesn't exist
if not os.path.exists(flattened_root):
    os.makedirs(flattened_root)

import shutil

# Iterate through each class directory in the original nested structure
for class_dir in os.listdir(test_DIR):
    class_path = os.path.join(test_DIR, class_dir)

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
test_dataset = datasets.ImageFolder(root=flattened_root, transform=transform)

val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)




# =============================================================================
def test(model, test_loader, device):
    model.eval()
    model.to(device)
    
    male_acc=0
    female_acc=0
    child_acc=0
    middle_acc=0
    senior_acc=0
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy for each class
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_correct = sum(class_correct)
    overall_total = sum(class_total)
    overall_accuracy = 100 * overall_correct / overall_total

    print("Accuracy for each class:")
    for i, class_name in enumerate(test_loader.dataset.classes):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name}: {class_acc:.2f}%")
        if(i<3):
            male_acc +=class_acc
        if(i>3):
            female_acc += class_acc
        if(i%3 == 0):
            child_acc += class_acc
        if(i%3 ==1 ):
            middle_acc += class_acc 
        if(i*3 ==2):
            senior_acc += class_acc
            
    print("-----------------------------------------------------")

    print(f"male Test Accuracy: {male_acc/3:.2f}%")
    print(f"female Test Accuracy: {female_acc/3:.2f}%") 
    print(f"child Test Accuracy: {child_acc/2:.2f}%")
    print(f"middle_age Test Accuracy: {middle_acc/2:.2f}%")
    print(f"senior Test Accuracy: {senior_acc/2:.2f}%")

    print("-----------------------------------------------------")

    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    return overall_accuracy
# =============================================================================

def calculate_precision(model, test_loader, device):
    model.eval()
    model.to(device)
    
    male_perc=0
    female_perc=0
    child_perc=0
    middle_perc=0
    senior_perc=0
    
    class_correct = [0] * len(test_loader.dataset.classes)
    class_predicted = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Calculate precision for each class
            for i in range(len(labels)):
                label = labels[i].item()
                prediction = predicted[i].item()
                if prediction == label:
                    class_correct[label] += 1
                class_predicted[prediction] += 1

    overall_correct = sum(class_correct)
    overall_predicted = sum(class_predicted)

    overall_precision = 100 * overall_correct / overall_predicted

    print("Precision for each class:")
    for i, class_name in enumerate(test_loader.dataset.classes):        
        if class_predicted[i] ==0 :
            class_predicted[i]=1
        class_precision = 100 * class_correct[i] / class_predicted[i]
        print(f"{class_name}: {class_precision:.2f}%")

        if(i<3):
            male_perc +=class_precision
        if(i>3):
            female_perc += class_precision
        if(i%3 == 0):
            child_perc += class_precision
        if(i%3 ==1 ):
            middle_perc += class_precision 
        if(i*3 ==2):
            senior_perc += class_precision
            
    print("-----------------------------------------------------")

    print(f"male Test precision: {male_perc/3:.2f}%")
    print(f"female Test precision: {female_perc/3:.2f}%") 
    print(f"child Test precision: {child_perc/2:.2f}%")
    print(f"middle_age Test precision: {middle_perc/2:.2f}%")
    print(f"senior Test precision: {senior_perc/2:.2f}%")

    print("-----------------------------------------------------")



    print(f"Overall Test Precision: {overall_precision:.2f}%")
    return overall_precision
# =============================================================================
def calculate_recall(model, test_loader, device):
    model.eval()
    model.to(device)

    male_rec=0
    female_rec=0
    child_rec=0
    middle_rec=0
    senior_rec=0


    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Calculate recall for each class
            for i in range(len(labels)):
                label = labels[i].item()
                prediction = predicted[i].item()
                if prediction == label:
                    class_correct[label] += 1
                class_total[label] += 1

    overall_correct = sum(class_correct)
    overall_total = sum(class_total)
    
    overall_recall = 100 * overall_correct / overall_total

    print("Recall for each class:")
    for i, class_name in enumerate(test_loader.dataset.classes):
        class_recall = 100 * class_correct[i] / class_total[i]
        print(f"{class_name}: {class_recall:.2f}%")

        if(i<3):
            male_rec +=class_recall
        if(i>3):
            female_rec += class_recall
        if(i%3 == 0):
            child_rec += class_recall
        if(i%3 ==1 ):
            middle_rec += class_recall 
        if(i*3 ==2):
            senior_rec += class_recall
            
    print("-----------------------------------------------------")

    print(f"male Test recall: {male_rec/3:.2f}%")
    print(f"female Test recall: {female_rec/3:.2f}%") 
    print(f"child Test recall: {child_rec/2:.2f}%")
    print(f"middle_age Test recall: {middle_rec/2:.2f}%")
    print(f"senior Test recall: {senior_rec/2:.2f}%")

    print("-----------------------------------------------------")


    print(f"Overall Test Recall: {overall_recall:.2f}%")
    return overall_recall
# =============================================================================

from sklearn.metrics import f1_score

def calculate_f1_score(model, test_loader, device):
    model.eval()
    model.to(device)
    
    male_f1=0
    female_f1=0
    child_f1=0
    middle_f1=0
    senior_f1=0

    class_correct = [0] * len(test_loader.dataset.classes)
    class_predicted = [0] * len(test_loader.dataset.classes)
    class_actual = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Calculate F1 score for each class
            for i in range(len(labels)):
                label = labels[i].item()
                prediction = predicted[i].item()
                if prediction == label:
                    class_correct[label] += 1
                class_predicted[prediction] += 1
                class_actual[label] += 1

    class_f1_scores = []
    print("F1 Score for each class:")
    for i, class_name in enumerate(test_loader.dataset.classes):
        precision = class_correct[i] / class_predicted[i] if class_predicted[i] > 0 else 0
        recall = class_correct[i] / class_actual[i] if class_actual[i] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        class_f1_scores.append(f1)
        print(f"{class_name}: {f1:.2f}")
       
        if(i<3):
            male_f1 +=f1
        if(i>3):
            female_f1 += f1
        if(i%3 == 0):
            child_f1 += f1
        if(i%3 ==1 ):
            middle_f1 += f1 
        if(i*3 ==2):
            senior_f1 += f1
            
    print("-----------------------------------------------------")

    print(f"male Test f1: {male_f1/3:.2f}%")
    print(f"female Test f1: {female_f1/3:.2f}%") 
    print(f"child Test f1: {child_f1/2:.2f}%")
    print(f"middle_age Test f1: {middle_f1/2:.2f}%")
    print(f"senior Test f1: {senior_f1/2:.2f}%")

    print("-----------------------------------------------------")

    overall_f1_score = f1_score(class_actual, class_predicted, average='weighted')

    print(f"Overall Test F1 Score: {overall_f1_score:.2f}")
    return overall_f1_score
# =============================================================================
def main():
    class_names = test_dataset.classes
    print(class_names)

    torch.manual_seed(42)
    weights_path = r"weights/best_1.pt"

    # Create a new model model (ensure it matches the architecture of your saved weights)
    model =  New(num_classes).to(device)

    
    # Load the saved weights into the model
    model_state_dict = torch.load(weights_path)
    model.load_state_dict(model_state_dict)

    # Calculate fold validation accuracy and accumulate total
    print("==============================================================")
    overall_acc= test(model, val_loader, device)
    print("==============================================================")
    overall_perc= calculate_precision(model, val_loader, device)
    print("==============================================================")
    overall_rec= calculate_recall(model, val_loader, device)
    print("==============================================================")
    overall_f1= calculate_f1_score(model, val_loader, device)
    print("==============================================================")







if __name__ == "__main__":
    main()                         