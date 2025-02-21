import os
import torch
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from MiniImagenet import MiniImagenet  # Ensure correct import path
from meta import Meta  # Import the MAML model

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train MAML on Mini-ImageNet and fine-tune on a custom dataset")
parser.add_argument("--miniimagenet_path", type=str, default="/content/miniimagenet", help="Path to Mini-ImageNet dataset")
parser.add_argument("--brain_tumor_path", type=str, default="/content/brain_tumor", help="Path to Brain Tumor dataset")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and testing")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--model_save_path", type=str, default="./maml_miniimagenet.pth", help="Path to save trained model")
args = parser.parse_args()

# Ensure dataset folders exist
if not os.path.exists(args.miniimagenet_path):
    raise FileNotFoundError(f"Mini-ImageNet dataset not found at {args.miniimagenet_path}")
if not os.path.exists(args.brain_tumor_path):
    raise FileNotFoundError(f"Brain Tumor dataset not found at {args.brain_tumor_path}")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Mini-ImageNet dataset for training
miniimagenet = MiniImagenet(args.miniimagenet_path, mode='train', n_way=5, k_shot=1, k_query=5, batchsz=100, resize=84)
train_loader = DataLoader(miniimagenet, batch_size=args.batch_size, shuffle=True)

# Define image transformations for Brain Tumor dataset
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # Resize to match Mini-ImageNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Brain Tumor dataset for fine-tuning
test_dataset = datasets.ImageFolder(args.brain_tumor_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize MAML model
maml_model = Meta(args).to(device)  # Ensure your MAML model is correctly imported
optimizer = torch.optim.Adam(maml_model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Store metrics for visualization
train_losses, test_accuracies, test_f1_scores, test_precisions, test_recalls = [], [], [], [], []

# Train MAML on Mini-ImageNet
print("Training MAML on Mini-ImageNet...")
for epoch in range(args.epochs):
    running_loss = 0
    for images, labels, _, _, _ in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = maml_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_losses[-1]:.4f}")

# Save trained model
torch.save(maml_model.state_dict(), args.model_save_path)
print(f"Model saved to {args.model_save_path}")

# Load the trained model for fine-tuning
print("Fine-tuning MAML on Brain Tumor dataset...")
maml_model.load_state_dict(torch.load(args.model_save_path))
maml_model.eval()

# Evaluate model on Brain Tumor dataset
true_labels, pred_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = maml_model(images)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="binary")
precision = precision_score(true_labels, pred_labels, average="binary")
recall = recall_score(true_labels, pred_labels, average="binary")

test_accuracies.append(accuracy)
test_f1_scores.append(f1)
test_precisions.append(precision)
test_recalls.append(recall)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Plot Training Loss Over Epochs
plt.figure(figsize=(10,5))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot Accuracy Over Epochs
plt.figure(figsize=(10,5))
plt.plot(range(1, len(test_accuracies)+1), test_accuracies, marker='o', linestyle='-', color='g', label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()
