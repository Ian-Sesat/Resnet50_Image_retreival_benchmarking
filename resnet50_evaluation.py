"""
ResNet50 Best Model — Full Training with Attention Head
========================================================
Extracts embeddings from best ResNet50 model and evaluates on:
- Image Retrieval : mAP, Recall@1, Recall@5
"""

import os
import numpy as np
import warnings
import faiss

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.cuda.amp import autocast

from sklearn.model_selection import train_test_split
from PIL import ImageFile
from tqdm import tqdm

# CONFIG 
DATA_DIR      = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'
SAVE_DIR      = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'
CHECKPOINT    = os.path.join(SAVE_DIR, 'best_model_fulltrain.pth')
EMBEDDINGS    = os.path.join(SAVE_DIR, 'embeddings_resnet50_best.npz')

NUM_CLASSES    = 100
BATCH_SIZE     = 16
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}

# Run flags
RUN_EXTRACTION = True
RUN_EVALUATION = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


# 1. TRANSFORMS 
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 2. DATASET 
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


def get_dataloaders(batch_size):
    clean_data = FilteredImageFolder(
        root=DATA_DIR,
        transform=val_transform,
        is_valid_file=lambda path: os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )

    indices = list(range(len(clean_data)))
    labels  = [clean_data.targets[i] for i in indices]

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, _     = train_test_split(
        train_val_idx, test_size=0.17647, stratify=train_val_labels, random_state=42
    )

    train_dataset = Subset(clean_data, train_idx)
    test_dataset  = Subset(clean_data, test_idx)

    print(f"Train : {len(train_dataset)} | Test : {len(test_dataset)}")

    train_loader = DataLoader(SafeDataset(train_dataset), batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SafeDataset(test_dataset),  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=4, pin_memory=True)

    return train_loader, test_loader


# 3. ATTENTION HEAD 
class AttentionHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=100):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_embedding=False):
        attn      = self.attention(x)
        x_att     = x * attn
        embedding = self.pool(x_att).squeeze()

        if return_embedding:
            return embedding

        out = self.dropout(embedding)
        return self.classifier(out)


# 4. MODEL 
class ResNet50WithAttention(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone       = backbone
        self.attention_head = AttentionHead(in_features=2048, num_classes=num_classes)

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        features = features.view(features.size(0), 2048, 7, 7)
        return self.attention_head(features, return_embedding)


def build_model():
    backbone         = models.resnet50(weights=None)
    backbone.avgpool = nn.Identity()
    backbone.fc      = nn.Identity()
    model            = ResNet50WithAttention(backbone, NUM_CLASSES)
    return model


# 5. EMBEDDING EXTRACTION 
def extract_embeddings(loader, model, device):
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)

            with autocast():
                embeddings = model(images, return_embedding=True)

            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels,     axis=0)
    )


# 6. RETRIEVAL EVALUATION 
def build_faiss_index(embeddings):
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors ")
    return index


def evaluate_map(test_embeddings, test_labels, index, train_labels):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    aps        = []
    chunk_size = 1000

    for start in tqdm(range(0, len(query), chunk_size), desc="mAP Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, index.ntotal)

        for i, idx in enumerate(indices):
            query_label    = test_labels[start + i]
            retrieved      = train_labels[idx]
            total_relevant = (train_labels == query_label).sum()

            ap      = 0.0
            correct = 0

            for rank, label in enumerate(retrieved, 1):
                if label == query_label:
                    correct += 1
                    ap      += correct / rank

            ap = ap / total_relevant if total_relevant > 0 else 0.0
            aps.append(ap)

    map_score = np.mean(aps)
    print(f"  mAP : {map_score:.4f}")
    return map_score


def evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    correct    = 0
    chunk_size = 10000

    for start in tqdm(range(0, len(query), chunk_size), desc=f"Recall@{k} Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label  = test_labels[start + i]
            top_k_labels = train_labels[idx]
            if query_label in top_k_labels:
                correct += 1

    recall = correct / len(test_embeddings)
    print(f"  Recall@{k} : {recall:.4f}")
    return recall


# MAIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
print(f"Model        : ResNet50 Full Training + Attention Head")

# Build and load model
model = build_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Model loaded from checkpoint ")

# Get dataloaders
train_loader, test_loader = get_dataloaders(BATCH_SIZE)

# EMBEDDING EXTRACTION

if RUN_EXTRACTION:
    print("\n" + "="*60)
    print("EXTRACTION — ResNet50 Best Model")
    print("="*60)

    torch.cuda.empty_cache()

    print("Extracting train embeddings ...")
    train_embeddings, train_labels = extract_embeddings(train_loader, model, device)
    print("Extracting test embeddings ...")
    test_embeddings,  test_labels  = extract_embeddings(test_loader,  model, device)

    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")

    np.savez(EMBEDDINGS,
             train_embeddings=train_embeddings, train_labels=train_labels,
             test_embeddings=test_embeddings,   test_labels=test_labels)
    print(f"Embeddings saved → {EMBEDDINGS}")

else:
    data             = np.load(EMBEDDINGS)
    train_embeddings = data['train_embeddings']
    train_labels     = data['train_labels']
    test_embeddings  = data['test_embeddings']
    test_labels      = data['test_labels']
    print("Embeddings loaded!")


# EVALUATION

if RUN_EVALUATION:
    print("\n" + "="*60)
    print("EVALUATION — ResNet50 Best Model")
    print("="*60)

    torch.cuda.empty_cache()
    index = build_faiss_index(train_embeddings)

    print("\n── Image Retrieval ──")
    map_score = evaluate_map(test_embeddings, test_labels, index, train_labels)
    recall_1  = evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1)
    recall_5  = evaluate_recall(test_embeddings, test_labels, index, train_labels, k=5)

    print("\n" + "="*60)
    print("FINAL RESULTS — ResNet50 Full Training + Attention")
    print("="*60)
    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 35)
    print(f"{'mAP':<25} {map_score*100:>9.2f}%")
    print(f"{'Recall@1':<25} {recall_1*100:>9.2f}%")
    print(f"{'Recall@5':<25} {recall_5*100:>9.2f}%")
    print("-" * 35)
    print("\n ALL DONE! ")