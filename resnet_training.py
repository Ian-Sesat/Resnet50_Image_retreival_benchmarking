"""
ResNet50 Three Model Comparison Script
=======================================
Compares three ResNet50 training approaches:
Model 1 → Last layer only (fc only)
Model 2 → Layer4 + dropout (partial training)
Model 3 → Full training (all layers)

Evaluates all three on:
- Image Retrieval  : Precision@1, Precision@5, kNN@21
- Classification   : Top-1, Top-5 (direct from attention head)
"""

import os
import numpy as np
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
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import train_test_split
from PIL import ImageFile
from tqdm import tqdm

# CONFIG 
DATA_DIR  = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'
SAVE_DIR  = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'

# Model 1 — Last layer only
M1_CHECKPOINT = os.path.join(SAVE_DIR, 'best_model_lastlayer.pth')
M1_EMBEDDINGS = os.path.join(SAVE_DIR, 'embeddings_lastlayer.npz')
M1_EPOCHS     = 25
M1_LR         = 0.001
M1_PATIENCE   = 3

# Model 2 — Layer4 + dropout
M2_CHECKPOINT = os.path.join(SAVE_DIR, 'best_model_layer4.pth')
M2_EMBEDDINGS = os.path.join(SAVE_DIR, 'embeddings_layer4.npz')
M2_EPOCHS     = 25
M2_LR         = 0.0005
M2_PATIENCE   = 5

# Model 3 — Full training
M3_CHECKPOINT = os.path.join(SAVE_DIR, 'best_model_fulltrain.pth')
M3_EMBEDDINGS = os.path.join(SAVE_DIR, 'embeddings_fulltrain.npz')
M3_EPOCHS     = 40
M3_LR         = 0.00005
M3_PATIENCE   = 5

# Focal Loss settings
FOCAL_ALPHA = 1
FOCAL_GAMMA = 2

NUM_CLASSES    = 100
BATCH_SIZE     = 64
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}

# Run flags
RUN_M1_TRAINING   = True
RUN_M1_EXTRACTION = True
RUN_M2_TRAINING   = True
RUN_M2_EXTRACTION = True
RUN_M3_TRAINING   = True
RUN_M3_EXTRACTION = True
RUN_EVALUATION    = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


# 1. TRANSFORMS 
# Model 1 and 2 — basic transforms
basic_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Model 3 — full augmentation
full_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Val/Test — same for all
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 2. FOCAL LOSS 
class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard/minority class examples.

    pt high (model confident + correct) → loss down-weighted → easy examples ignored
    pt low  (model confused)            → loss stays high    → hard examples focused
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss    = self.ce_loss(inputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


# 3. ATTENTION HEAD 
class AttentionHead(nn.Module):
    """
    Attention Head for ResNet50.

    Replaces the Global Average Pool + fc layer.
    Learns which spatial regions of the 7×7 feature map matter most.

    Training  : forward(x)                    → class scores
    Retrieval : forward(x, return_embedding)  → 2048 attention-weighted embeddings
    """
    def __init__(self, in_features=2048, num_classes=100):
        super().__init__()

        # Learns a score (0-1) for each of the 49 spatial locations (7×7)
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, 1, kernel_size=1),  # 2048 → 1 attention map
            nn.Sigmoid()                                # values between 0 and 1
        )

        self.pool       = nn.AdaptiveAvgPool2d(1)      # pool after attention
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_embedding=False):
        """
        x shape: (B, 2048, 7, 7) ← output of ResNet50 backbone

        Step 1: compute attention map  → (B, 1, 7, 7)
        Step 2: multiply with features → (B, 2048, 7, 7) attended
        Step 3: pool attended features → (B, 2048) embedding
        Step 4: classify               → (B, num_classes)
        """
        attn      = self.attention(x)           # (B, 1, 7, 7)
        x_att     = x * attn                    # (B, 2048, 7, 7) attended features
        embedding = self.pool(x_att).squeeze()  # (B, 2048) attention-weighted embedding

        if return_embedding:
            return embedding                    # ← used for retrieval

        out = self.dropout(embedding)
        return self.classifier(out)             # ← used for classification


# 4. RESNET50 WITH ATTENTION 
def build_resnet50(freeze_mode='none'):
    """
    Build ResNet50 with AttentionHead.

    freeze_mode:
        'none'   → all layers trainable (Model 3)
        'layer4' → freeze everything except layer4 + attention head (Model 2)
        'fc'     → freeze everything except attention head only (Model 1)
    """
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Remove original avgpool and fc — we replace with AttentionHead
    backbone.avgpool = nn.Identity()
    backbone.fc      = nn.Identity()

    # Full model: backbone + attention head
    class ResNet50WithAttention(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone      = backbone
            self.attention_head = AttentionHead(
                in_features=2048,
                num_classes=num_classes
            )

        def forward(self, x, return_embedding=False):
            # Get 2048 × 7 × 7 feature map from backbone
            features = self.backbone(x)
            # Reshape from (B, 2048*7*7) back to (B, 2048, 7, 7)
            features = features.view(features.size(0), 2048, 7, 7)
            return self.attention_head(features, return_embedding)

    model = ResNet50WithAttention(backbone, NUM_CLASSES)

    # Apply freezing
    if freeze_mode == 'fc':
        # Freeze everything except attention head
        for name, param in model.named_parameters():
            if 'attention_head' not in name:
                param.requires_grad = False

    elif freeze_mode == 'layer4':
        # Freeze everything except layer4 and attention head
        for name, param in model.named_parameters():
            if 'attention_head' not in name and 'layer4' not in name:
                param.requires_grad = False

    # freeze_mode == 'none' → all layers trainable

    return model


#  5. DATASET 
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


def get_dataloaders(train_transform, batch_size):
    augmented_data = FilteredImageFolder(
        root=DATA_DIR,
        transform=train_transform,
        is_valid_file=lambda path: os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )
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
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.17647, stratify=train_val_labels, random_state=42
    )

    train_dataset = Subset(augmented_data, train_idx)
    val_dataset   = Subset(clean_data,     val_idx)
    test_dataset  = Subset(clean_data,     test_idx)

    print(f"Train : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    train_loader = DataLoader(SafeDataset(train_dataset), batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_skip_none,
                              num_workers=20, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(SafeDataset(val_dataset),   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=20, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(SafeDataset(test_dataset),  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_skip_none,
                              num_workers=20, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader


# 6. EARLY STOPPING 
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience  = patience
        self.counter   = 0
        self.best_loss = float('inf')
        self.stop      = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


# 7. TRAINING 
def train_model(model, train_loader, val_loader, epochs, lr,
                checkpoint_path, weight_decay=1e-4, patience=5):

    criterion      = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer      = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler         = GradScaler()
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss  = float('inf')

    for epoch in range(epochs):

        # ── Training ──
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                predictions = model(images)
                loss        = criterion(predictions, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── Validation ──
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

                with autocast():
                    predictions = model(images)
                    loss        = criterion(predictions, labels)

                val_loss += loss.item()
                correct  += (predictions.argmax(1) == labels).sum().item()
                total    += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc  = correct / total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        early_stopping(val_loss)
        if early_stopping.stop:
            print("Early stopping triggered!")
            break


# 8. EMBEDDING EXTRACTION 
def extract_embeddings(loader, model, device):
    """Extract 2048 attention-weighted embeddings."""
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)

            with autocast():
                embeddings = model(images, return_embedding=True)  # 2048 attended

            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels,     axis=0)
    )


# 9. RETRIEVAL EVALUATION 
def build_faiss_index(embeddings):
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def evaluate_retrieval(test_embeddings, test_labels, index, train_labels, k=5):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    p_at_1     = []
    p_at_k     = []
    chunk_size = 10000

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

    p1 = np.mean(p_at_1)
    p5 = np.mean(p_at_k)
    print(f"  Precision@1  : {p1:.4f}")
    print(f"  Precision@{k} : {p5:.4f}")
    return p1, p5


def knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    correct    = 0
    chunk_size = 10000

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
    print(f"  kNN Accuracy (k={k}) : {accuracy:.4f}")
    return accuracy


# 10. CLASSIFICATION EVALUATION 
def evaluate_classification(model, test_loader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total        = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Classification Evaluation"):
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            with autocast():
                predictions = model(images)

            correct_top1  += (predictions.argmax(1) == labels).sum().item()
            top5_pred      = predictions.topk(5, dim=1).indices
            correct_top5  += sum(
                labels[i] in top5_pred[i] for i in range(len(labels))
            )
            total += labels.size(0)

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    print(f"  Top-1 Accuracy : {top1:.4f}")
    print(f"  Top-5 Accuracy : {top5:.4f}")
    return top1, top5


# MAIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
print(f"Focal Loss   : alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
print(f"AMP          : enabled ")
print(f"Attention    : enabled ")



# MODEL 1 — LAST LAYER ONLY

print("\n" + "="*60)
print("MODEL 1 — Last Layer Only (Attention Head)")
print("="*60)

train_loader_m1, val_loader_m1, test_loader_m1 = get_dataloaders(basic_train_transform, BATCH_SIZE)

model1 = build_resnet50(freeze_mode='fc').to(device)
print(f"Trainable params : {sum(p.numel() for p in model1.parameters() if p.requires_grad):,}")

if RUN_M1_TRAINING:
    print("\nTraining Model 1 ...")
    train_model(model1, train_loader_m1, val_loader_m1,
                M1_EPOCHS, M1_LR, M1_CHECKPOINT,
                weight_decay=1e-4, patience=M1_PATIENCE)
else:
    model1.load_state_dict(torch.load(M1_CHECKPOINT, map_location=device))
    print("Model 1 loaded from checkpoint!")

if RUN_M1_EXTRACTION:
    print("\nExtracting Model 1 embeddings ...")
    model1.load_state_dict(torch.load(M1_CHECKPOINT, map_location=device))
    model1.eval()
    train_emb_m1, train_lbl_m1 = extract_embeddings(train_loader_m1, model1, device)
    val_emb_m1,   val_lbl_m1   = extract_embeddings(val_loader_m1,   model1, device)
    test_emb_m1,  test_lbl_m1  = extract_embeddings(test_loader_m1,  model1, device)
    np.savez(M1_EMBEDDINGS,
             train_embeddings=train_emb_m1, train_labels=train_lbl_m1,
             val_embeddings=val_emb_m1,     val_labels=val_lbl_m1,
             test_embeddings=test_emb_m1,   test_labels=test_lbl_m1)
    print("Model 1 embeddings saved!")
else:
    data         = np.load(M1_EMBEDDINGS)
    train_emb_m1 = data['train_embeddings']
    train_lbl_m1 = data['train_labels']
    val_emb_m1   = data['val_embeddings']
    val_lbl_m1   = data['val_labels']
    test_emb_m1  = data['test_embeddings']
    test_lbl_m1  = data['test_labels']
    print("Model 1 embeddings loaded!")

# MODEL 2 — LAYER4 + DROPOUT

print("\n" + "="*60)
print("MODEL 2 — Layer4 + Dropout (Attention Head)")
print("="*60)

train_loader_m2, val_loader_m2, test_loader_m2 = get_dataloaders(basic_train_transform, BATCH_SIZE)

model2 = build_resnet50(freeze_mode='layer4').to(device)
print(f"Trainable params : {sum(p.numel() for p in model2.parameters() if p.requires_grad):,}")

if RUN_M2_TRAINING:
    print("\nTraining Model 2 ...")
    train_model(model2, train_loader_m2, val_loader_m2,
                M2_EPOCHS, M2_LR, M2_CHECKPOINT,
                weight_decay=1e-4, patience=M2_PATIENCE)
else:
    model2.load_state_dict(torch.load(M2_CHECKPOINT, map_location=device))
    print("Model 2 loaded from checkpoint!")

if RUN_M2_EXTRACTION:
    print("\nExtracting Model 2 embeddings ...")
    model2.load_state_dict(torch.load(M2_CHECKPOINT, map_location=device))
    model2.eval()
    train_emb_m2, train_lbl_m2 = extract_embeddings(train_loader_m2, model2, device)
    val_emb_m2,   val_lbl_m2   = extract_embeddings(val_loader_m2,   model2, device)
    test_emb_m2,  test_lbl_m2  = extract_embeddings(test_loader_m2,  model2, device)
    np.savez(M2_EMBEDDINGS,
             train_embeddings=train_emb_m2, train_labels=train_lbl_m2,
             val_embeddings=val_emb_m2,     val_labels=val_lbl_m2,
             test_embeddings=test_emb_m2,   test_labels=test_lbl_m2)
    print("Model 2 embeddings saved!")
else:
    data         = np.load(M2_EMBEDDINGS)
    train_emb_m2 = data['train_embeddings']
    train_lbl_m2 = data['train_labels']
    val_emb_m2   = data['val_embeddings']
    val_lbl_m2   = data['val_labels']
    test_emb_m2  = data['test_embeddings']
    test_lbl_m2  = data['test_labels']
    print("Model 2 embeddings loaded!")


# MODEL 3 — FULL TRAINING

print("\n" + "="*60)
print("MODEL 3 — Full Training (Attention Head)")
print("="*60)

train_loader_m3, val_loader_m3, test_loader_m3 = get_dataloaders(full_train_transform, BATCH_SIZE)

model3 = build_resnet50(freeze_mode='none').to(device)
print(f"Trainable params : {sum(p.numel() for p in model3.parameters() if p.requires_grad):,}")

if RUN_M3_TRAINING:
    print("\nTraining Model 3 ...")
    train_model(model3, train_loader_m3, val_loader_m3,
                M3_EPOCHS, M3_LR, M3_CHECKPOINT,
                weight_decay=1e-4, patience=M3_PATIENCE)
else:
    model3.load_state_dict(torch.load(M3_CHECKPOINT, map_location=device))
    print("Model 3 loaded from checkpoint!")

if RUN_M3_EXTRACTION:
    print("\nExtracting Model 3 embeddings ...")
    model3.load_state_dict(torch.load(M3_CHECKPOINT, map_location=device))
    model3.eval()
    train_emb_m3, train_lbl_m3 = extract_embeddings(train_loader_m3, model3, device)
    val_emb_m3,   val_lbl_m3   = extract_embeddings(val_loader_m3,   model3, device)
    test_emb_m3,  test_lbl_m3  = extract_embeddings(test_loader_m3,  model3, device)
    np.savez(M3_EMBEDDINGS,
             train_embeddings=train_emb_m3, train_labels=train_lbl_m3,
             val_embeddings=val_emb_m3,     val_labels=val_lbl_m3,
             test_embeddings=test_emb_m3,   test_labels=test_lbl_m3)
    print("Model 3 embeddings saved!")
else:
    data         = np.load(M3_EMBEDDINGS)
    train_emb_m3 = data['train_embeddings']
    train_lbl_m3 = data['train_labels']
    val_emb_m3   = data['val_embeddings']
    val_lbl_m3   = data['val_labels']
    test_emb_m3  = data['test_embeddings']
    test_lbl_m3  = data['test_labels']
    print("Model 3 embeddings loaded!")


# EVALUATION
if RUN_EVALUATION:

    # ── Model 1 ──
    print("\n" + "="*60)
    print("MODEL 1 — Last Layer Only — Evaluation")
    print("="*60)
    index_m1     = build_faiss_index(train_emb_m1)
    p1_m1, p5_m1 = evaluate_retrieval(test_emb_m1, test_lbl_m1, index_m1, train_lbl_m1, k=5)
    knn_m1       = knn_evaluation(test_emb_m1, test_lbl_m1, index_m1, train_lbl_m1, k=21)
    model1.load_state_dict(torch.load(M1_CHECKPOINT, map_location=device))
    top1_m1, top5_m1 = evaluate_classification(model1, test_loader_m1, device)

    # ── Model 2 ──
    print("\n" + "="*60)
    print("MODEL 2 — Layer4 + Dropout — Evaluation")
    print("="*60)
    index_m2     = build_faiss_index(train_emb_m2)
    p1_m2, p5_m2 = evaluate_retrieval(test_emb_m2, test_lbl_m2, index_m2, train_lbl_m2, k=5)
    knn_m2       = knn_evaluation(test_emb_m2, test_lbl_m2, index_m2, train_lbl_m2, k=21)
    model2.load_state_dict(torch.load(M2_CHECKPOINT, map_location=device))
    top1_m2, top5_m2 = evaluate_classification(model2, test_loader_m2, device)

    # ── Model 3 ──
    print("\n" + "="*60)
    print("MODEL 3 — Full Training — Evaluation")
    print("="*60)
    index_m3     = build_faiss_index(train_emb_m3)
    p1_m3, p5_m3 = evaluate_retrieval(test_emb_m3, test_lbl_m3, index_m3, train_lbl_m3, k=5)
    knn_m3       = knn_evaluation(test_emb_m3, test_lbl_m3, index_m3, train_lbl_m3, k=21)
    model3.load_state_dict(torch.load(M3_CHECKPOINT, map_location=device))
    top1_m3, top5_m3 = evaluate_classification(model3, test_loader_m3, device)

    # ── Final Comparison Table ──
    print("\n" + "="*75)
    print("FINAL COMPARISON TABLE")
    print("(Focal Loss + AMP + Attention Head)")
    print("="*75)
    print(f"\n{'Metric':<25} {'M1 (Last Layer)':>18} {'M2 (Layer4)':>18} {'M3 (Full)':>18}")
    print("-" * 79)
    print(f"{'Precision@1':<25} {p1_m1*100:>17.2f}% {p1_m2*100:>17.2f}% {p1_m3*100:>17.2f}%")
    print(f"{'Precision@5':<25} {p5_m1*100:>17.2f}% {p5_m2*100:>17.2f}% {p5_m3*100:>17.2f}%")
    print(f"{'kNN Accuracy @21':<25} {knn_m1*100:>17.2f}% {knn_m2*100:>17.2f}% {knn_m3*100:>17.2f}%")
    print(f"{'Top-1 Accuracy':<25} {top1_m1*100:>17.2f}% {top1_m2*100:>17.2f}% {top1_m3*100:>17.2f}%")
    print(f"{'Top-5 Accuracy':<25} {top5_m1*100:>17.2f}% {top5_m2*100:>17.2f}% {top5_m3*100:>17.2f}%")
    print("-" * 79)
    print("\n ALL DONE! ")
