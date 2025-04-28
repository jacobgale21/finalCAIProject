#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CAI4104 -- Project -- eval.py

This file contains the evaluation code to load a trained model from a specified checkpoint, evaluate it on a test dataset, and report performance. We will run this script on the test dataset to evaluate the model's performance.
use the command:

python eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"

Note that you need to write the model creation function and call it in the load_trained_model function. You may also
need to change the predict function (e.g., if your pipeline is not compatible with the provided implementation).
Please test the model creation function and the model loading function and predict function to make sure they work. If we cannot load or evaluate your model, we will apply a penalty.

"""

import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

######### Functions #########

def load_test_dataset(data_dir: str, batch_size: int, num_workers: int, image_size: int):
    """
    Loads the test dataset from a given directory. The directory must contain subfolders
    for each class (like in training). Applies only evaluation transforms.

    Args:
        data_dir (str): Path to test data folder.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        image_size (int): Desired image size.

    Returns:
        DataLoader: DataLoader object for the test dataset.
    """
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader


def create_resnet18_model(num_classes):

    model = models.resnet18()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def create_cnn(num_classes):
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding="same")  # Input: 3 channels, Output: 16 filters
            #self.batch1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
            #self.batch2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
            #self.batch3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, 3, padding="same")
            #self.batch4 = nn.BatchNorm2d(256)
            self.conv5 = nn.Conv2d(256, 512, 3, padding="same")
            #self.batch5 = nn.BatchNorm2d(512)
            #224 -> 112 -> 56 -> 28 -> 14 - >7
            self.pool = nn.MaxPool2d(2, 2)
            
            self.fc1 = nn.Linear(512 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            
            x = x.view(x.size(0), -1) # flatten
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x

    model = CNN(num_classes=num_classes)
                          
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    model.optimizer = optimizer
    model.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)

    return model


    model = CNN(num_classes=num_classes)
                          
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
    model.optimizer = optimizer
    model.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)

    return model

def load_trained_model(model_path: str, num_classes: int, device: str, image_size: int = 224):
    """
    Builds your model architecture, adjusts the classification head to
    the given number of classes, and loads the trained model weights from a local file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        num_classes (int): Number of output classes. (Should be 12 but left for consistency.)
        device (str): Device for model loading ('cuda' or 'cpu').
        image_size (int): desired input image size. (Not used here but kept for consistency.)

    Returns:
        model: The model loaded on device. (If you are not using pytorch nn.Module directly, it is fine but make sure what it loads is compatible with the rest of the code.)
    """

    if 'cnn' in model_path.lower():
        model = create_cnn(num_classes)
        print("Using custom CNN model architecture.")
    else:
        model = create_resnet18_model(num_classes)
        print("Using ResNet18 model architecture.")
    
    ## Change/rewrite the rest of the function as needed, but make sure what it outputs works with the other functions (e.g., predict)

    # Load local state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move the model to the specified device and set evaluation mode.
    model = model.to(device)
    model.eval()
    return model


def predict(model, x):
    """
    Computes the predicted labels for a batch of input images.

    Args:
        model: Your trained model
        x (torch.Tensor): Input batch of images.

    Returns:
        torch.Tensor: Predicted labels.
    """
    
    ## Change/rewrite the function as needed, but make sure it outputs predictions in a way that it works with the rest of the code
    
    with torch.no_grad():
        outputs = model(x)
        _, y_pred = torch.max(outputs, 1)
    return y_pred


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model: Your trained model
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device used for evaluation.

    Returns:
        tuple: (test_loss, test_accuracy, per_class_accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    correct = 0
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(len(labels)):
                class_total[labels[i]] += 1
                if preds[i] == labels[i]:
                    class_correct[labels[i]] += 1
    test_loss = running_loss / total
    test_accuracy = correct / total
    per_class_accuracy = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(len(class_correct))]

    return test_loss, test_accuracy, per_class_accuracy



######### Main() #########

if __name__ == "__main__":
    exit_code = 0

    parser = argparse.ArgumentParser(description="Eval script for CAI4104 project")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="Path(s) to the trained model checkpoint(s) (e.g., models/trained_model.pth best_cnn_model.pt)")
    parser.add_argument("--test_data", type=str, default="project_test_data",
                        help="Directory containing the test dataset with class subfolders")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--group_id", type=int, required=True, help="Project Group ID (non-negative integer)")
    parser.add_argument("--project_title", type=str, required=True, help="Project Title (at least 4 characters)")

    args = parser.parse_args()

    project_group_id = args.group_id
    project_title = args.project_title

    assert project_group_id >= 0, "Group ID must be non-negative"
    assert len(project_title) >= 4, "Project title must be at least 4 characters long"

    st = time.time()

    print('\n---------- [Eval] (Project: {}, Group: {}) ---------'.format(project_title, project_group_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluation device:", device)

    test_loader = load_test_dataset(args.test_data, args.batch_size, args.num_workers, args.image_size)
    
    num_classes = len(test_loader.dataset.classes)
    print("Number of classes:", num_classes)

    # Evaluate each model path
    for model_path in args.model_paths:
        print(f"\n=== Evaluating model: {model_path} ===")
        try:
            model = load_trained_model(model_path, num_classes, device, args.image_size)
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            continue

        # Most Occurring Class Baseline
        class_counts = [0] * num_classes
        for label in test_loader.dataset.targets:
            class_counts[label] += 1
        most_occurring_class = np.argmax(class_counts)
        most_occurring_class_accuracy = class_counts[most_occurring_class] / len(test_loader.dataset.targets)
        print("Most Occurring Class Classifier Accuracy: {:.2f}%".format(most_occurring_class_accuracy * 100))

        # Evaluate model
        test_loss, test_accuracy, per_class_accuracy  = evaluate_model(model, test_loader, device)
        print("Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(test_loss, test_accuracy * 100))

    et = time.time()
    elapsed = et - st
    print('\n---------- [Eval] Elapsed time -- total: {:.1f} seconds ---------\n'.format(elapsed))

    sys.exit(exit_code)

