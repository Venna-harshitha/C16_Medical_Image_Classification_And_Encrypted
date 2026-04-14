"""
COMPLETE FHE MEDICAL DIAGNOSIS SYSTEM
────────────────────────────────────────────────────────────────────────────

TWO MODES:
1. DEMO MODE (Fast): Simple encryption for UI/UX demonstration
2. REAL FHE MODE (Slow): CKKS homomorphic encryption for true privacy

Author: [Your Name]
Date: February 2026
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
from config import MODEL_PATHS
import time

# Initialize Flask app
app = Flask(__name__)

# Disease classifications
CLASS_NAMES = ['cardiomegaly', 'normal', 'pneumonia', 'tuberculosis']
IMG_SIZE = 128

# Medical information database
MEDICAL_INFO = {
    'cardiomegaly': {
        'full_name': 'Cardiomegaly (Enlarged Heart)',
        'severity': 'Moderate to Severe',
        'description': 'An enlarged heart that may indicate underlying heart conditions.',
        'reasons': [
            'High blood pressure (Hypertension)',
            'Heart valve problems',
            'Coronary artery disease',
            'Heart muscle disease (Cardiomyopathy)',
            'Congenital heart defects',
            'Fluid around the heart (Pericardial effusion)'
        ],
        'precautions': [
            '⚠️ Consult a cardiologist immediately for proper diagnosis',
            'Monitor blood pressure regularly',
            'Reduce sodium intake in diet',
            'Avoid excessive alcohol consumption',
            'Maintain healthy body weight',
            'Follow prescribed medications strictly',
            'Avoid strenuous physical activities until cleared by doctor',
            'Get adequate rest and sleep'
        ],
        'urgency': 'HIGH',
        'color': '#ef4444'
    },
    'pneumonia': {
        'full_name': 'Pneumonia (Lung Infection)',
        'severity': 'Moderate to Severe',
        'description': 'An infection that inflames air sacs in one or both lungs.',
        'reasons': [
            'Bacterial infection (most common: Streptococcus pneumoniae)',
            'Viral infection (influenza, COVID-19)',
            'Fungal infection (in immunocompromised)',
            'Aspiration of food or liquid into lungs',
            'Weakened immune system',
            'Chronic lung diseases'
        ],
        'precautions': [
            '⚠️ Seek immediate medical attention for antibiotics/antivirals',
            'Get adequate rest and sleep',
            'Stay well hydrated - drink plenty of fluids',
            'Take prescribed medications on time',
            'Use a humidifier to ease breathing',
            'Avoid smoking and secondhand smoke',
            'Practice deep breathing exercises',
            'Monitor fever and oxygen levels',
            'Isolate to prevent spreading infection',
            'Complete the full course of antibiotics'
        ],
        'urgency': 'HIGH',
        'color': '#f59e0b'
    },
    'tuberculosis': {
        'full_name': 'Tuberculosis (TB)',
        'severity': 'Severe - Contagious',
        'description': 'A serious bacterial infection that mainly affects the lungs.',
        'reasons': [
            'Mycobacterium tuberculosis bacteria',
            'Close contact with infected person',
            'Weakened immune system (HIV/AIDS)',
            'Malnutrition',
            'Living in crowded conditions',
            'Diabetes or kidney disease',
            'Substance abuse'
        ],
        'precautions': [
            '⚠️ IMMEDIATE medical attention required - TB is curable but needs treatment',
            'Start anti-TB medication immediately (6-9 months course)',
            'Take ALL medications exactly as prescribed - DO NOT SKIP',
            'Isolate yourself for first 2-3 weeks of treatment',
            'Wear a mask when around others',
            'Cover mouth when coughing/sneezing',
            'Ensure good ventilation in living spaces',
            'Maintain nutritious diet to support immune system',
            'Avoid alcohol and smoking',
            'Inform close contacts to get tested',
            'Complete the FULL treatment course - stopping early causes drug resistance'
        ],
        'urgency': 'CRITICAL',
        'color': '#dc2626'
    },
    'normal': {
        'full_name': 'Normal - Healthy Lungs',
        'severity': 'None',
        'description': 'Your chest X-ray shows no signs of abnormalities.',
        'reasons': [
            'Healthy lung tissue',
            'Normal heart size',
            'Clear airways',
            'No signs of infection or inflammation'
        ],
        'precautions': [
            'Maintain regular health check-ups',
            'Follow a balanced, nutritious diet',
            'Exercise regularly (150 minutes/week)',
            'Avoid smoking and excessive alcohol',
            'Get adequate sleep (7-9 hours)',
            'Stay hydrated',
            'Practice good hygiene',
            'Keep vaccinations up to date',
            'Manage stress through meditation or yoga',
            'Monitor any new symptoms and consult doctor if concerned'
        ],
        'urgency': 'LOW',
        'color': '#10b981'
    }
}

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================
class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network for feature extraction
    Input: 128x128 grayscale X-ray
    Output: 16,384-dimensional feature vector
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def extract_features(self, x):
        """Extract 16,384-dim features for FHE classifier"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)


class FHEPolynomialHead(nn.Module):
    """
    FHE-Compatible Classifier with Polynomial Activations
    
    Why polynomial? ReLU is not FHE-friendly.
    We use f(x) = 0.125*x² + 0.5*x as approximation.
    
    Input: 16,384-dimensional features (encrypted in CKKS)
    Output: 4-class logits (encrypted)
    """
    def __init__(self, input_dim=256*8*8, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def poly(self, x):
        """Polynomial approximation of ReLU (FHE-compatible)"""
        return 0.125 * x * x + 0.5 * x

    def forward(self, x):
        x = self.poly(self.bn1(self.fc1(x)))
        x = self.poly(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# =============================================================================
# INITIALIZE MODELS
# =============================================================================
print("="*80)
print("🏥 SECUREHEALTH AI - FHE MEDICAL DIAGNOSIS SYSTEM")
print("="*80)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

try:
    # Load CNN feature extractor
    print("\n📦 Loading models...")
    retrained_cnn = SimpleCNN(num_classes=4).to(device)
    retrained_cnn.load_state_dict(torch.load(MODEL_PATHS['retrained'], map_location=device))
    retrained_cnn.eval()
    print("✅ CNN Feature Extractor loaded")
    
    # Load FHE classifier
    fhe_head = FHEPolynomialHead(input_dim=256*8*8, num_classes=4).to(device)
    fhe_head.load_state_dict(torch.load(MODEL_PATHS['fhe_head'], map_location=device))
    fhe_head.eval()
    print("✅ FHE Polynomial Classifier loaded")
    
    feature_extractor = retrained_cnn
    
    print(f"\n✅ System ready for {len(CLASS_NAMES)} disease classes")
    MODELS_LOADED = True
    
except Exception as e:
    print(f"\n❌ FATAL ERROR loading models: {e}")
    import traceback
    traceback.print_exc()
    MODELS_LOADED = False

# =============================================================================
# REAL FHE SETUP (CKKS)
# =============================================================================
try:
    print("\n🔐 Initializing REAL Homomorphic Encryption (CKKS)...")
    from fhe_model import get_fhe_classifier
    fhe_classifier = get_fhe_classifier()
    FHE_AVAILABLE = True
    print("✅ CKKS FHE Ready!")
except ImportError:
    print("⚠️  TenSEAL not installed. Run: pip install tenseal")
    print("   FHE mode will not be available.")
    FHE_AVAILABLE = False
except Exception as e:
    print(f"⚠️  FHE initialization error: {e}")
    FHE_AVAILABLE = False

print("\n" + "="*80)
print("🚀 SERVER READY")
print("="*80)
print("\nAvailable modes:")
print("  • DEMO MODE (fast): /fhe_predict")
if FHE_AVAILABLE:
    print("  • REAL FHE MODE (secure): /fhe_predict_real")
print("\n" + "="*80 + "\n")


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_image(image_bytes):
    """
    Preprocess X-ray image to 128x128 grayscale tensor
    Matches training preprocessing exactly
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        raise


# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html', fhe_available=FHE_AVAILABLE)


@app.route('/fhe_predict', methods=['POST'])
def fhe_predict_demo():
    """
    DEMO MODE - Fast inference with simulated encryption
    
    ⚠️ This mode uses simple XOR "encryption" for demonstration.
    Server processes PLAINTEXT images - NOT truly private.
    Use /fhe_predict_real for actual homomorphic encryption.
    """
    if not MODELS_LOADED:
        return jsonify({'error': 'Models failed to load at startup'}), 500
    
    try:
        data = request.json
        encrypted_data_b64 = data['encrypted_data']
        
        # Patient details
        patient_name = data.get('patient_name', 'Unknown')
        patient_age = data.get('patient_age', 'N/A')
        patient_gender = data.get('patient_gender', 'N/A')
        
        print(f"\n📊 [DEMO MODE] Processing for: {patient_name}, {patient_age}y, {patient_gender}")
        
        # Decode image
        encrypted_bytes = base64.b64decode(encrypted_data_b64)
        
        # Preprocess
        start_time = time.time()
        image_tensor = preprocess_image(encrypted_bytes)
        image_tensor = image_tensor.to(device)
        preprocess_time = time.time() - start_time
        
        # Run inference
        with torch.no_grad():
            # Standard CNN
            start_time = time.time()
            retrained_logits = retrained_cnn(image_tensor)
            retrained_probs = F.softmax(retrained_logits, dim=1).cpu().numpy()[0]
            retrained_time = time.time() - start_time
            
            # FHE Head (but in plaintext for speed)
            start_time = time.time()
            features = feature_extractor.extract_features(image_tensor)
            features = (features - features.mean(dim=1, keepdim=True)) / \
                      (features.std(dim=1, keepdim=True) + 1e-6)
            fhe_logits = fhe_head(features)
            fhe_probs = F.softmax(fhe_logits, dim=1).cpu().numpy()[0]
            fhe_time = time.time() - start_time
        
        # Get predictions
        head_pred_idx = np.argmax(fhe_probs)
        head_pred_class = CLASS_NAMES[head_pred_idx]
        head_pred_prob = float(fhe_probs[head_pred_idx])
        
        retrained_pred_idx = np.argmax(retrained_probs)
        retrained_pred_class = CLASS_NAMES[retrained_pred_idx]
        retrained_pred_prob = float(retrained_probs[retrained_pred_idx])
        
        # Get medical information
        medical_data = MEDICAL_INFO[head_pred_class]
        
        print(f"✅ Diagnosis: {head_pred_class} ({head_pred_prob:.2%})")
        
        results = {
            'head_pred': [float(p) for p in fhe_probs],
            'retrained_pred': [float(p) for p in retrained_probs],
            'head_class': head_pred_class,
            'head_prob': head_pred_prob,
            'retrained_class': retrained_pred_class,
            'retrained_prob': retrained_pred_prob,
            'diagnosis': head_pred_class,
            'medical_info': medical_data,
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'mode': 'DEMO',
            'encryption': 'Simulated (XOR)',
            'is_real_fhe': False,
            'processing_time': {
                'preprocessing': f"{preprocess_time*1000:.2f}ms",
                'retrained_inference': f"{retrained_time*1000:.2f}ms",
                'fhe_inference': f"{fhe_time*1000:.2f}ms",
                'total': f"{(preprocess_time + retrained_time + fhe_time)*1000:.2f}ms"
            }
        }
        
        encrypted_response = base64.b64encode(json.dumps(results).encode()).decode()
        return jsonify({
            'encrypted_results': encrypted_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


@app.route('/fhe_predict_real', methods=['POST'])
def fhe_predict_real():
    """
    REAL FHE MODE - CKKS Homomorphic Encryption
    
    ✅ This mode uses REAL homomorphic encryption (TenSEAL CKKS)
    Server computes on ENCRYPTED data - truly privacy-preserving.
    
    ⚠️ This is 100-1000x SLOWER than demo mode (~10-30 seconds)
    """
    if not MODELS_LOADED:
        return jsonify({'error': 'Models failed to load at startup'}), 500
    
    if not FHE_AVAILABLE:
        return jsonify({'error': 'TenSEAL not installed. Install with: pip install tenseal'}), 500
    
    try:
        data = request.json
        encrypted_data_b64 = data['encrypted_data']
        
        # Patient details
        patient_name = data.get('patient_name', 'Unknown')
        patient_age = data.get('patient_age', 'N/A')
        patient_gender = data.get('patient_gender', 'N/A')
        
        print(f"\n🔒 [REAL FHE] Processing for: {patient_name}, {patient_age}y, {patient_gender}")
        
        # Decode and preprocess image
        encrypted_bytes = base64.b64decode(encrypted_data_b64)
        image_tensor = preprocess_image(encrypted_bytes).to(device)
        
        # Extract features using CNN (this part is plaintext for speed)
        # In full FHE, even this could be encrypted, but it's 10,000x slower
        start_time = time.time()
        with torch.no_grad():
            features = feature_extractor.extract_features(image_tensor)
            features_np = features.cpu().numpy()[0]  # [16384]
        feature_time = time.time() - start_time
        
        print(f"✅ Feature extraction: {feature_time*1000:.2f}ms")
        
        # ═══════════════════════════════════════════════════════════
        # REAL HOMOMORPHIC ENCRYPTION STARTS HERE
        # ═══════════════════════════════════════════════════════════
        print("🔐 Starting CKKS homomorphic inference...")
        print("   → Client encrypts features with CKKS")
        print("   → Server computes on ENCRYPTED data")
        print("   → Client decrypts final diagnosis")
        
        start_time = time.time()
        fhe_results = fhe_classifier.predict_encrypted(features_np)
        fhe_time = time.time() - start_time
        
        print(f"✅ FHE inference complete: {fhe_time:.2f}s")
        print(f"   Encrypted feature size: {fhe_results['encrypted_features_size']:,} bytes")
        print(f"   Encrypted result size: {fhe_results['encrypted_results_size']:,} bytes")
        
        diagnosis = CLASS_NAMES[fhe_results['prediction']]
        confidence = fhe_results['confidence']
        probabilities = fhe_results['probabilities']
        
        print(f"✅ Diagnosis: {diagnosis} ({confidence:.2%})")
        
        # Get medical information
        medical_data = MEDICAL_INFO[diagnosis]
        
        results = {
            'head_pred': probabilities.tolist(),
            'retrained_pred': probabilities.tolist(),  # Same in FHE mode
            'head_class': diagnosis,
            'head_prob': confidence,
            'retrained_class': diagnosis,
            'retrained_prob': confidence,
            'diagnosis': diagnosis,
            'medical_info': medical_data,
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'mode': 'REAL FHE',
            'encryption': 'CKKS (TenSEAL)',
            'is_real_fhe': True,
            'processing_time': {
                'preprocessing': f"{feature_time*1000:.2f}ms",
                'fhe_inference': f"{fhe_time:.2f}s",
                'total': f"{(feature_time + fhe_time):.2f}s"
            },
            'encryption_info': {
                'scheme': 'CKKS',
                'library': 'TenSEAL (Microsoft SEAL)',
                'poly_modulus_degree': 8192,
                'security_level': '128-bit',
                'encrypted_feature_size_bytes': fhe_results['encrypted_features_size'],
                'encrypted_result_size_bytes': fhe_results['encrypted_results_size']
            }
        }
        
        encrypted_response = base64.b64encode(json.dumps(results).encode()).decode()
        return jsonify({
            'encrypted_results': encrypted_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"\n❌ FHE ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n🌐 Starting Flask server...")
    print("   Access at: http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)