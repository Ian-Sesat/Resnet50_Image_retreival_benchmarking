"""
ResNet50 Training + Embedding Extraction + Image Retrieval

A clean pipeline for:
1. Training ResNet50 on a custom image dataset
2. Extracting embeddings from the penultimate layer
3. Evaluating image retrieval with Precision@K and kNN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import faiss

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from PIL import ImageFile
from tqdm import tqdm

# CONFIG 

DATA_DIR       = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'   # path to your dataset folder
CHECKPOINT    = 'best_model_dropout.pth'  # where to save/load the best model
EMBEDDINGS     = 'embeddings_dropout.npz'   # where to save/load embeddings
PLOT_PATH      = 'training_plot.png' # where to save the training plot

NUM_CLASSES    = 100
BATCH_SIZE     = 64
EPOCHS         = 10
LEARNING_RATE  = 0.0005
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# TEST_RATIO   = 0.15  (remainder)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'} # List of system folders to ignore

# Set to True to train, False to skip training and load saved model
RUN_TRAINING   = False
# Set to True to extract embeddings, False to load saved embeddings
RUN_EXTRACTION = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# 1. DATA

# Image transforms — resize, convert to tensor, normalise
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def safe_loader(index):
    try:
        return dataset[index]
    except Exception:
        return None

class SafeDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            return self.subset[idx]
        except Exception:
            return None
        
def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [
            folder for folder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, folder))
            and folder not in IGNORE_FOLDERS
        ]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

# Load dataset using only valid folders
dataset = FilteredImageFolder(
    root=DATA_DIR,
    transform=transform,
    is_valid_file=lambda path: os.path.splitext(path)[1].lower() in IMAGE_EXTS
)
print(f"Classes found : {len(dataset.classes)}")
print(f"Total images  : {len(dataset)}")

# Stratified split — ensures each class is equally represented in each split
indices = list(range(len(dataset)))
labels  = [dataset.targets[i] for i in indices]

# Step 1: split off 15% for test
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=VAL_RATIO,
    stratify=labels,
    random_state=42
)

# Step 2: split remaining into train and val (0.176 of 85% ≈ 15% of total)
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.17647,
    stratify=train_val_labels,
    random_state=42
)

# Wrap indices into Subset datasets
train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, val_idx)
test_dataset  = Subset(dataset, test_idx)

print(f"Train : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

# Wrap in DataLoaders (feeds images in batches)
train_loader = DataLoader(SafeDataset(train_dataset), batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_skip_none,
                          num_workers=20, pin_memory=True)
val_loader   = DataLoader(SafeDataset(val_dataset),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_skip_none,
                          num_workers=20, pin_memory=True)
test_loader  = DataLoader(SafeDataset(test_dataset),  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_skip_none,
                          num_workers=20, pin_memory=True)
# 2. MODEL

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace last layer for 100 classes
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),        # randomly switches off 50% of neurons during training
    nn.Linear(2048, 100)
)

# Unfreeze fc
for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
print(f"Using device : {device}")


# 3. TRAINING

def train(model, train_loader, val_loader, epochs):
    """Train ResNet50 and save the best model based on validation loss."""

    criterion     = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler     = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=5,  # reduce lr every 5 epochs
                    gamma=0.1     # multiply lr by 0.1
                )
    best_val_loss = float('inf')

    # Track metrics for plotting
    history = {
        'train_loss': [],
        'val_loss'  : [],
        'val_acc'   : []
    }

    for epoch in range(epochs):

        #Training
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss        = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validating"):
                if batch is None:
                    continue
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)
                loss        = criterion(predictions, labels)

                val_loss += loss.item()
                correct  += (predictions.argmax(1) == labels).sum().item()
                total    += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc  = correct / total

        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  Best model saved (val_loss={best_val_loss:.4f})")

    return history

def plot_training(history):
    """Save a plot of training/validation loss and validation accuracy."""

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ResNet50 Training History', fontsize=14, fontweight='bold')

    #Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['val_acc'], 'g-o', label='Val Accuracy')
    ax2.set_title('Validation Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Training plot saved → {PLOT_PATH}")


if RUN_TRAINING:
    history = train(model, train_loader, val_loader, EPOCHS)
    plot_training(history)


# 4. EMBEDDING EXTRACTION

# Load best model
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Best model loaded!")

# Remove the last classification layer to get embeddings
embedding_model = nn.Sequential(*list(model.children())[:-1])
embedding_model = embedding_model.to(device)
print("Embedding model ready!")


def extract_embeddings(loader, device):
    """Pass images through the embedding model and return (embeddings, labels)."""

    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            embeddings = embedding_model(images)
            embeddings = embeddings.squeeze()  # (B, 2048, 1, 1) → (B, 2048)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels,     axis=0)
    )


if RUN_EXTRACTION:
    print("Extracting embeddings ...")
    train_embeddings, train_labels = extract_embeddings(train_loader, device)
    val_embeddings,   val_labels   = extract_embeddings(val_loader,   device)
    test_embeddings,  test_labels  = extract_embeddings(test_loader,  device)

    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Val embeddings   : {val_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")

    # Save to disk
    np.savez(EMBEDDINGS,
             train_embeddings=train_embeddings, train_labels=train_labels,
             val_embeddings=val_embeddings,     val_labels=val_labels,
             test_embeddings=test_embeddings,   test_labels=test_labels)
    print(f"Embeddings saved → {EMBEDDINGS}")

else:
    # Load from disk
    data             = np.load(EMBEDDINGS)
    train_embeddings = data['train_embeddings']
    train_labels     = data['train_labels']
    val_embeddings   = data['val_embeddings']
    val_labels       = data['val_labels']
    test_embeddings  = data['test_embeddings']
    test_labels      = data['test_labels']
    print(f"Embeddings loaded from {EMBEDDINGS}")


# 5. IMAGE RETRIEVAL EVALUATION
def build_faiss_index(embeddings):
    """Build a FAISS index from train embeddings."""
    embeddings = embeddings.astype('float32')
    
    # Normalise embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine similarity after normalisation
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def evaluate_retrieval(test_embeddings, test_labels, index, train_labels, k=5):
    query = test_embeddings.astype('float32')
    faiss.normalize_L2(query)

    p_at_1 = []
    p_at_k = []
    
    chunk_size = 1000  # search 1000 queries at a time

    for start in tqdm(range(0, len(query), chunk_size), desc="Precision Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        
        _, indices = index.search(chunk, k)
        
        for i, idx in enumerate(indices):
            query_label  = test_labels[start + i]
            top_k_labels = train_labels[idx]
            correct      = (top_k_labels == query_label)

            p_at_1.append(float(correct[0]))
            p_at_k.append(correct.sum() / k)

    print(f"Precision@1  : {np.mean(p_at_1):.4f}")
    print(f"Precision@{k} : {np.mean(p_at_k):.4f}")


def knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21):
    query = test_embeddings.astype('float32')
    faiss.normalize_L2(query)

    correct    = 0
    chunk_size = 1000

    for start in tqdm(range(0, len(query), chunk_size), desc="kNN Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]

        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label     = test_labels[start + i]
            top_k_labels    = train_labels[idx]
            votes           = np.bincount(top_k_labels, minlength=NUM_CLASSES)
            predicted_class = np.argmax(votes)

            if predicted_class == query_label:
                correct += 1

    accuracy = correct / len(test_embeddings)
    print(f"kNN Accuracy (k={k}) : {accuracy:.4f}")

print("\n── Retrieval Evaluation ──")
index = build_faiss_index(train_embeddings)
#evaluate_retrieval(test_embeddings, test_labels, index, train_labels, k=5)
knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21)