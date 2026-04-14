gpu_code.iypnb.py
"""
ENHANCED MEDICAL IMAGE CLASSIFIER - PyTorch GPU
================================================
Simple CNN with clear metrics output
"""

import numpy as np
import os
import cv2
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

# =====================================================================
# GPU SETUP
# =====================================================================
print("=" * 80)
print("ENHANCED MEDICAL CNN - PYTORCH GPU")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA: {torch.version.cuda}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    cudnn.benchmark = True
else:
    print("⚠ No GPU - using CPU")

print("=" * 80)

# Set seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# =====================================================================
# CONFIGURATION
# =====================================================================
TRAIN_DIR = r'A:\PROJECTC16\train'
TEST_DIR = r'A:\PROJECTC16\test'
OUTPUT_DIR = r'A:\PROJECTC16\output_pytorch'
IMG_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASS_NAMES = ['cardiomegaly', 'normal', 'pneumonia', 'tuberculosis']
class_to_num = {name: idx for idx, name in enumerate(CLASS_NAMES)}
num_classes = len(CLASS_NAMES)

print(f"\nConfig: {IMG_SIZE}px | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print(f"Classes: {CLASS_NAMES}")
print("=" * 80)

# =====================================================================
# DATASET
# =====================================================================
class MedicalDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        print(f"\nLoading from: {data_dir}")
        
        class_folders = [f for f in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, f))]
        
        for folder in class_folders:
            folder_path = os.path.join(data_dir, folder)
            print(f"  {folder}...", end=' ')
            count = 0
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                self.data.append([img, class_to_num[folder]])
                count += 1
            
            print(f"{count} images")
        
        shuffle(self.data)
        print(f"✓ Total: {len(self.data)} images\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img, label

# =====================================================================
# LOAD DATA
# =====================================================================
print("[1/5] Loading datasets...")
train_dataset_full = MedicalDataset(TRAIN_DIR)
test_dataset = MedicalDataset(TEST_DIR)

# Split train/val
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(
    train_dataset_full, [train_size, val_size], generator=generator
)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# =====================================================================
# MODEL
# =====================================================================
print("\n[2/5] Building CNN model...")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Dense
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.bn5(self.fc1(x))))
        x = self.drop6(F.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        return x

model = SimpleCNN(num_classes=num_classes).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model built! Parameters: {total_params:,}")

# =====================================================================
# TRAINING SETUP
# =====================================================================
print("\n[3/5] Setting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}',
                         'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), correct / total

# =====================================================================
# TRAIN
# =====================================================================
print("\n[4/5] Training on GPU...")
print("=" * 80)

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
patience_counter = 0
patience = 7

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 80)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    scheduler.step(val_acc)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train: Loss={train_loss:.4f} | Acc={train_acc*100:.2f}%")
    print(f"Val:   Loss={val_loss:.4f} | Acc={val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth'))
        print(f"✓ Best model saved! Val Acc: {val_acc*100:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\n✓ Training complete!")

# =====================================================================
# EVALUATE
# =====================================================================
print("\n[5/5] Evaluating on test set...")
print("=" * 80)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_pytorch.pth')))
model.eval()

all_preds = []
all_probs = []
y_test = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(outputs.max(1)[1].cpu().numpy())
        y_test.extend(labels.numpy())

predictions = np.vstack(all_probs)
y_pred = np.array(all_preds)
y_test = np.array(y_test)

# Calculate accuracies
train_accuracy = max(history['train_acc'])
val_accuracy = max(history['val_acc'])
test_accuracy = np.mean(y_pred == y_test)

# Metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# =====================================================================
# RESULTS
# =====================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS - PYTORCH GPU")
print("=" * 80)
print(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("-" * 80)
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print("=" * 80)

if test_accuracy >= 0.95:
    print("✅ EXCELLENT! >95% accuracy!")
elif test_accuracy >= 0.90:
    print("✅ GOOD! >90% accuracy!")

print("=" * 80)

print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# =====================================================================
# VISUALIZATIONS
# =====================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Accuracy
ax1 = plt.subplot(2, 3, 1)
epochs = range(1, len(history['train_acc']) + 1)
plt.plot(epochs, history['train_acc'], 'b-', linewidth=3, marker='o', markersize=5, label='Train')
plt.plot(epochs, history['val_acc'], 'r-', linewidth=3, marker='s', markersize=5, label='Val')
plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95%')
plt.fill_between(epochs, 0.95, 1.0, alpha=0.1, color='green')
plt.title('Model Accuracy (PyTorch GPU)', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.ylim(0, 1.05)

# 2. Loss
ax2 = plt.subplot(2, 3, 2)
plt.plot(epochs, history['train_loss'], 'b-', linewidth=3, marker='o', markersize=5, label='Train')
plt.plot(epochs, history['val_loss'], 'r-', linewidth=3, marker='s', markersize=5, label='Val')
plt.title('Model Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

# 3. Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=13)
plt.ylabel('True', fontsize=13)
plt.xticks(rotation=45, ha='right')

# 4. Normalized
ax4 = plt.subplot(2, 3, 4)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Normalized (%)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=13)
plt.ylabel('True', fontsize=13)
plt.xticks(rotation=45, ha='right')

# 5. Per-Class Accuracy
ax5 = plt.subplot(2, 3, 5)
class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = plt.bar(CLASS_NAMES, class_acc, color=colors, edgecolor='black', linewidth=3)
plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2.5)
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(alpha=0.3, axis='y')

for bar, acc in zip(bars, class_acc):
    h = bar.get_height()
    col = 'green' if h >= 0.95 else ('orange' if h >= 0.90 else 'red')
    plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h*100:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color=col)

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
status = "✅ >95%!" if test_accuracy >= 0.95 else ("✅ >90%" if test_accuracy >= 0.90 else "📊 GOOD")

summary = f"""
╔═══════════════════════════════════════╗
║    PYTORCH CNN - MEDICAL IMAGING      ║
╚═══════════════════════════════════════╝

STATUS: {status}

ACCURACIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Train:    {train_accuracy*100:6.2f}%
  Val:      {val_accuracy*100:6.2f}%
  Test:     {test_accuracy*100:6.2f}%

METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Precision:  {precision:.4f}
  Recall:     {recall:.4f}
  F1-Score:   {f1:.4f}

DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Train:      {len(train_dataset):5d}
  Val:        {len(val_dataset):5d}
  Test:       {len(test_dataset):5d}

HARDWARE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Device:     {gpu_name}

MODEL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Parameters: {total_params:,}
  Epochs:     {len(epochs)}
  Batch:      {BATCH_SIZE}
  Image:      {IMG_SIZE}x{IMG_SIZE}

ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ 4 Conv Blocks (32→64→128→256)
  ✓ Batch Normalization
  ✓ Dropout Regularization
  ✓ GPU Acceleration
"""

plt.text(0.05, 0.5, summary, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round',
                  facecolor='lightgreen' if test_accuracy >= 0.95 else 'lightyellow',
                  alpha=0.6, pad=1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pytorch_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Results saved!")

# Save history
np.save(os.path.join(OUTPUT_DIR, 'pytorch_history.npy'), history)

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - PYTORCH GPU!")
print("=" * 80)
print(f"Output: {OUTPUT_DIR}")
print(f"  • best_model_pytorch.pth")
print(f"  • pytorch_results.png")
print(f"  • pytorch_history.npy")
print("=" * 80)
print(f"\n🎯 FINAL: TRAIN {train_accuracy*100:.2f}% | TEST {test_accuracy*100:.2f}%")
if test_accuracy >= 0.95:
    print("🏆 EXCELLENT PERFORMANCE!")
print("=" * 80)