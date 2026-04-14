# Create this as: A:\FHE_web\debug_models.py
import torch

print("="*60)
print("DEBUGGING MODEL FILES")
print("="*60)

# Check FHE head
print("\n📦 FHE Head weights:")
fhe = torch.load(r'D:\D\FHE_web\models\fhe_head.pth', map_location='cpu', weights_only=True)
for key, value in fhe.items():
    print(f"  {key}: {value.shape}")

print("\n📦 Retrained model weights:")
retrained = torch.load(r'D:\D\FHE_web\models\retrained_model.pth', map_location='cpu', weights_only=True)
for key, value in retrained.items():
    print(f"  {key}: {value.shape}")

print("\n" + "="*60)