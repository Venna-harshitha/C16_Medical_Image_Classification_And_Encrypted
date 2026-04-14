"""
ACCURACY COMPARISON WITH PUBLICATION-QUALITY VISUALIZATION
==========================================================
✅ Plain CNN vs FHE Polynomial Classifier
✅ Publication-ready graphs for IEEE/Journal papers
✅ Multiple visualization styles included
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================= CONFIG =================
TEST_DIR = r"D:\PROJECTC16\test"
PLAIN_MODEL_PATH = r"D:\D\FHE_web\models\retrained_model.pth"
FHE_HEAD_PATH = r"D:\D\FHE_web\models\fhe_head.pth"
OUTPUT_DIR = r"D:\D\FHE_web\paper_visualizations"

IMG_SIZE = 128
FEATURE_DIM = 256 * 8 * 8
BATCH_SIZE = 64

CLASS_NAMES = ['cardiomegaly', 'normal', 'pneumonia', 'tuberculosis']
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================= MODELS =================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def extract_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)

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


class FHEPolynomialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 4)

    def poly(self, x):
        return 0.125 * x * x + 0.5 * x

    def forward(self, x):
        x = self.poly(self.bn1(self.fc1(x)))
        x = self.poly(self.bn2(self.fc2(x)))
        return self.fc3(x)


def disable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.p = 0.0


# ================= DATA =================
class MedicalDataset(Dataset):
    def __init__(self, root):
        self.data = []

        VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

        for c in CLASS_NAMES:
            class_dir = os.path.join(root, c)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(VALID_EXTENSIONS):
                    # 🚫 Skip encrypted / non-image files
                    continue

                img_path = os.path.join(class_dir, fname)
                self.data.append((img_path, class_to_idx[c]))

        print(f"Loaded {len(self.data)} valid test images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Unreadable image file: {path}")

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img, label


# ================= VISUALIZATION FUNCTIONS =================
def create_accuracy_comparison_bar(plain_acc, fhe_acc, output_path):
    """
    Main accuracy comparison bar chart - REQUIRED FOR PAPER
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Plain CNN', 'FHE Polynomial\nClassifier']
    accuracies = [plain_acc * 100, fhe_acc * 100]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', 
                   linewidth=2, alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add accuracy drop annotation
    drop = (plain_acc - fhe_acc) * 100
    ax.annotate(f'Privacy Cost\n↓ {drop:.2f}%',
                xy=(0.5, max(accuracies) - 2), xytext=(0.5, max(accuracies) + 3),
                ha='center', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Comparison: Plain CNN vs FHE-Compatible Classifier', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Threshold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_per_class_comparison(plain_preds, fhe_preds, labels, output_path):
    """
    Per-class accuracy comparison
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    plain_class_acc = []
    fhe_class_acc = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        mask = labels == i
        plain_class_acc.append((plain_preds[mask] == labels[mask]).mean() * 100)
        fhe_class_acc.append((fhe_preds[mask] == labels[mask]).mean() * 100)
    
    bars1 = ax.bar(x - width/2, plain_class_acc, width, label='Plain CNN', 
                   color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fhe_class_acc, width, label='FHE Polynomial', 
                   color='#A23B72', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Disease Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_confusion_matrices(plain_preds, fhe_preds, labels, output_path):
    """
    Side-by-side confusion matrices
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_plain = confusion_matrix(labels, plain_preds)
    cm_fhe = confusion_matrix(labels, fhe_preds)
    
    # Plain CNN confusion matrix
    sns.heatmap(cm_plain, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, ax=axes[0],
                annot_kws={'size': 11, 'weight': 'bold'})
    axes[0].set_title('Plain CNN', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    
    # FHE confusion matrix
    sns.heatmap(cm_fhe, annot=True, fmt='d', cmap='Purples', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, ax=axes[1],
                annot_kws={'size': 11, 'weight': 'bold'})
    axes[1].set_title('FHE Polynomial Classifier', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_metrics_comparison(plain_preds, fhe_preds, labels, output_path):
    """
    Multi-metric comparison (Accuracy, Precision, Recall, F1)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    plain_metrics = {
        'Accuracy': accuracy_score(labels, plain_preds) * 100,
        'Precision': precision_score(labels, plain_preds, average='weighted') * 100,
        'Recall': recall_score(labels, plain_preds, average='weighted') * 100,
        'F1-Score': f1_score(labels, plain_preds, average='weighted') * 100
    }
    
    fhe_metrics = {
        'Accuracy': accuracy_score(labels, fhe_preds) * 100,
        'Precision': precision_score(labels, fhe_preds, average='weighted') * 100,
        'Recall': recall_score(labels, fhe_preds, average='weighted') * 100,
        'F1-Score': f1_score(labels, fhe_preds, average='weighted') * 100
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(plain_metrics))
    width = 0.35
    
    plain_values = list(plain_metrics.values())
    fhe_values = list(fhe_metrics.values())
    
    bars1 = ax.bar(x - width/2, plain_values, width, label='Plain CNN',
                   color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fhe_values, width, label='FHE Polynomial',
                   color='#A23B72', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plain_metrics.keys(), fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_combined_paper_figure(plain_acc, fhe_acc, plain_preds, fhe_preds, labels, output_path):
    """
    Combined figure with all visualizations - PERFECT FOR PAPERS
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main accuracy comparison (large, top center)
    ax1 = fig.add_subplot(gs[0, :2])
    methods = ['Plain CNN', 'FHE Polynomial\nClassifier']
    accuracies = [plain_acc * 100, fhe_acc * 100]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    drop = (plain_acc - fhe_acc) * 100
    ax1.text(0.5, max(accuracies) - 3, f'Privacy Cost: {drop:.2f}%',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Summary statistics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = f"""
    RESULTS SUMMARY
    ═══════════════════
    
    Plain CNN:
      Accuracy: {plain_acc*100:.2f}%
    
    FHE Polynomial:
      Accuracy: {fhe_acc*100:.2f}%
    
    Privacy Cost:
      Drop: {drop:.2f}%
    
    Status:
      {'✅ Both >90%' if plain_acc >= 0.90 and fhe_acc >= 0.90 else '⚠ Needs tuning'}
    
    Test Samples: {len(labels)}
    """
    ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 3. Per-class accuracy (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    plain_class_acc = []
    fhe_class_acc = []
    for i in range(len(CLASS_NAMES)):
        mask = labels == i
        plain_class_acc.append((plain_preds[mask] == labels[mask]).mean() * 100)
        fhe_class_acc.append((fhe_preds[mask] == labels[mask]).mean() * 100)
    
    bars1 = ax3.bar(x - width/2, plain_class_acc, width, label='Plain CNN', color='#2E86AB')
    bars2 = ax3.bar(x + width/2, fhe_class_acc, width, label='FHE Polynomial', color='#A23B72')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Disease Class', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Per-Class Performance', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Confusion matrices (bottom row)
    ax4 = fig.add_subplot(gs[2, 0:2])
    cm_plain = confusion_matrix(labels, plain_preds)
    cm_fhe = confusion_matrix(labels, fhe_preds)
    
    # Show FHE confusion matrix
    sns.heatmap(cm_fhe, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, ax=ax4, annot_kws={'size': 10})
    ax4.set_title('FHE Polynomial - Confusion Matrix', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Actual', fontsize=10, fontweight='bold')
    
    # 5. Metrics comparison (bottom right)
    ax5 = fig.add_subplot(gs[2, 2])
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics_names = ['Prec', 'Rec', 'F1']
    plain_vals = [
        precision_score(labels, plain_preds, average='weighted') * 100,
        recall_score(labels, plain_preds, average='weighted') * 100,
        f1_score(labels, plain_preds, average='weighted') * 100
    ]
    fhe_vals = [
        precision_score(labels, fhe_preds, average='weighted') * 100,
        recall_score(labels, fhe_preds, average='weighted') * 100,
        f1_score(labels, fhe_preds, average='weighted') * 100
    ]
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    ax5.bar(x_pos - width/2, plain_vals, width, label='Plain', color='#2E86AB')
    ax5.bar(x_pos + width/2, fhe_vals, width, label='FHE', color='#A23B72')
    ax5.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
    ax5.set_title('Metrics', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metrics_names, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.set_ylim(0, 110)
    ax5.grid(axis='y', alpha=0.3)
    
    plt.suptitle('FHE-Compatible Medical Image Classification - Complete Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


# ================= EVALUATION =================
def evaluate():
    print("\n" + "="*80)
    print("ACCURACY COMPARISON WITH PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*80)
    
    # Load dataset
    dataset = MedicalDataset(TEST_DIR)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Load models
    print("\n📂 Loading models...")
    cnn = SimpleCNN().to(device)
    cnn.load_state_dict(torch.load(PLAIN_MODEL_PATH, map_location=device, weights_only=False))
    disable_dropout(cnn)
    cnn.eval()
    
    fhe_head = FHEPolynomialHead().to(device)
    fhe_head.load_state_dict(torch.load(FHE_HEAD_PATH, map_location=device, weights_only=False))
    fhe_head.eval()
    print("✅ Models loaded")

    plain_preds, fhe_preds, labels = [], [], []

    print("\n🔄 Evaluating...")
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing"):
            x = x.to(device)
            
            plain_out = cnn(x)
            
            feats = cnn.extract_features(x)
            feats = (feats - feats.mean(1, keepdim=True)) / (feats.std(1, keepdim=True) + 1e-6)
            fhe_out = fhe_head(feats)

            plain_preds.extend(plain_out.argmax(1).cpu().numpy())
            fhe_preds.extend(fhe_out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    plain_preds = np.array(plain_preds)
    fhe_preds = np.array(fhe_preds)
    labels = np.array(labels)
    
    plain_acc = (plain_preds == labels).mean()
    fhe_acc = (fhe_preds == labels).mean()

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Plain CNN:        {plain_acc*100:.2f}%")
    print(f"FHE Polynomial:   {fhe_acc*100:.2f}%")
    print(f"Accuracy Drop:    {(plain_acc-fhe_acc)*100:.2f}%")
    print("="*80)

    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    create_accuracy_comparison_bar(
        plain_acc, fhe_acc, 
        os.path.join(OUTPUT_DIR, '1_accuracy_comparison_bar.png')
    )
    
    create_per_class_comparison(
        plain_preds, fhe_preds, labels,
        os.path.join(OUTPUT_DIR, '2_per_class_comparison.png')
    )
    
    create_confusion_matrices(
        plain_preds, fhe_preds, labels,
        os.path.join(OUTPUT_DIR, '3_confusion_matrices.png')
    )
    
    create_metrics_comparison(
        plain_preds, fhe_preds, labels,
        os.path.join(OUTPUT_DIR, '4_metrics_comparison.png')
    )
    
    create_combined_paper_figure(
        plain_acc, fhe_acc, plain_preds, fhe_preds, labels,
        os.path.join(OUTPUT_DIR, '5_combined_paper_figure.png')
    )
    
    print("\n✅ ALL VISUALIZATIONS SAVED!")
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. accuracy_comparison_bar.png       - Main comparison (REQUIRED)")
    print("  2. per_class_comparison.png          - Per-disease analysis")
    print("  3. confusion_matrices.png            - Confusion matrices")
    print("  4. metrics_comparison.png            - Multi-metric view")
    print("  5. combined_paper_figure.png         - All-in-one figure")
    
    print("\n📄 For your paper, use:")
    print("  • IEEE/Conference: '1_accuracy_comparison_bar.png' or '5_combined_paper_figure.png'")
    print("  • Journal: '5_combined_paper_figure.png' (comprehensive)")
    print("  • Presentation: All figures")
    
    print("\n🎉 COMPLETE!")


if __name__ == "__main__":
    evaluate()