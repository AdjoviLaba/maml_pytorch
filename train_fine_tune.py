import os
import random
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from maml import MAML  # Ensure you have a proper import path to your MAML model
from PIL import Image

# Argument parser to handle dataset paths and hyperparameters
parser = argparse.ArgumentParser(description="Train MAML on Mini-ImageNet and fine-tune on a custom dataset")
parser.add_argument("--miniimagenet_path", type=str, default="./miniimagenet", help="Path to Mini-ImageNet dataset")
parser.add_argument("--brain_tumor_path", type=str, default="./brain_tumor", help="Path to Brain Tumor dataset")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and testing")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--model_save_path", type=str, default="./maml_miniimagenet.pth", help="Path to save the trained model")
args = parser.parse_args()

# Ensure dataset folders exist
if not os.path.exists(args.miniimagenet_path):
    raise FileNotFoundError(f"Mini-ImageNet dataset not found at {args.miniimagenet_path}")
if not os.path.exists(args.brain_tumor_path):
    raise FileNotFoundError(f"Brain Tumor dataset not found at {args.brain_tumor_path}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # Resize to match Mini-ImageNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Mini-ImageNet dataset for training
train_dataset = datasets.ImageFolder(args.miniimagenet_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Load Brain Tumor dataset for testing
test_dataset = datasets.ImageFolder(args.brain_tumor_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize MAML model
maml_model = MAML()
optimizer = torch.optim.Adam(maml_model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Store metrics for visualization
train_losses = []
test_accuracies = []
test_f1_scores = []
test_precisions = []
test_recalls = []

# Train MAML on Mini-ImageNet
print("Training MAML on Mini-ImageNet...")
for epoch in range(args.epochs):
    running_loss = 0
    for images, labels in train_loader:
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
        outputs = maml_model(images)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        pred_labels.extend(preds.numpy())

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

# Plot training loss over epochs
plt.figure(figsize=(10,5))
plt.plot(range(1, args.epochs+1), train_losses, marker='o', linestyle='-', color='b', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Display sample images of normal and tumor cases
def show_sample_images():
    normal_dir = os.path.join(args.brain_tumor_path, "normal")
    tumor_dir = os.path.join(args.brain_tumor_path, "tumor")

    normal_images = [os.path.join(normal_dir, img) for img in os.listdir(normal_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    tumor_images = [os.path.join(tumor_dir, img) for img in os.listdir(tumor_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    sample_normal = random.choice(normal_images)
    sample_tumor = random.choice(tumor_images)

    img_normal = Image.open(sample_normal)
    img_tumor = Image.open(sample_tumor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_normal)
    axes[0].set_title("Normal Brain")
    axes[0].axis("off")
    axes[1].imshow(img_tumor)
    axes[1].set_title("Brain with Tumor")
    axes[1].axis("off")

    plt.show()

show_sample_images()
