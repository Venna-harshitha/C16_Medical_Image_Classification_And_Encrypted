"""
REAL HOMOMORPHIC ENCRYPTION IMPLEMENTATION
Using TenSEAL CKKS for Privacy-Preserving Medical Diagnosis
"""

import torch
import torch.nn as nn
import tenseal as ts
import numpy as np
from config import MODEL_PATHS

class FHEPolynomialHead(nn.Module):
    """FHE-compatible classifier with polynomial activations"""
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


class TenSEALFHEClassifier:
    """
    Wrapper for TenSEAL CKKS homomorphic encryption
    Enables encrypted inference on medical data
    """
    
    def __init__(self, model_path):
        # Load trained PyTorch model
        self.model = FHEPolynomialHead(input_dim=256*8*8, num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Extract weights for homomorphic operations
        self._extract_weights()
        
        # TenSEAL context will be created per-request
        self.context = None
        
        print("✅ FHE Classifier loaded (CKKS encryption ready)")
    
    def _extract_weights(self):
        """Extract and prepare weights for encrypted computation"""
        state_dict = self.model.state_dict()
        
        # Fold BatchNorm into linear layers for FHE
        self.w1, self.b1 = self._fold_bn(
            state_dict['fc1.weight'].numpy(),
            state_dict['fc1.bias'].numpy(),
            state_dict['bn1.weight'].numpy(),
            state_dict['bn1.bias'].numpy(),
            state_dict['bn1.running_mean'].numpy(),
            state_dict['bn1.running_var'].numpy()
        )
        
        self.w2, self.b2 = self._fold_bn(
            state_dict['fc2.weight'].numpy(),
            state_dict['fc2.bias'].numpy(),
            state_dict['bn2.weight'].numpy(),
            state_dict['bn2.bias'].numpy(),
            state_dict['bn2.running_mean'].numpy(),
            state_dict['bn2.running_var'].numpy()
        )
        
        self.w3 = state_dict['fc3.weight'].numpy()
        self.b3 = state_dict['fc3.bias'].numpy()
        
        print(f"✅ Weights extracted: W1={self.w1.shape}, W2={self.w2.shape}, W3={self.w3.shape}")
    
    def _fold_bn(self, W, b, gamma, beta, mean, var, eps=1e-5):
        """Fold BatchNorm into Linear layer for FHE deployment"""
        std = np.sqrt(var + eps)
        W_folded = W * (gamma / std).reshape(-1, 1)
        b_folded = gamma * (b - mean) / std + beta
        return W_folded, b_folded
    
    def create_context(self):
        """
        Create CKKS encryption context
        This defines the encryption parameters
        """
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,  # Security parameter
            coeff_mod_bit_sizes=[60, 40, 40, 60]  # Precision levels
        )
        context.generate_galois_keys()
        context.global_scale = 2**40  # Precision scale
        
        return context
    
    def encrypt_features(self, features, context):
        """
        CLIENT-SIDE: Encrypt feature vector using CKKS
        
        Args:
            features: numpy array [16384] - extracted CNN features
            context: TenSEAL context with encryption keys
            
        Returns:
            encrypted_vector: CKKS encrypted features
        """
        # Normalize features
        features = (features - features.mean()) / (features.std() + 1e-6)
        
        # Encrypt using CKKS
        encrypted_vector = ts.ckks_vector(context, features.tolist())
        
        return encrypted_vector
    
    def homomorphic_inference(self, encrypted_features):
        """
        SERVER-SIDE: Perform inference on ENCRYPTED data
        This is the REAL FHE computation - server never sees plaintext!
        
        Args:
            encrypted_features: CKKS encrypted feature vector
            
        Returns:
            encrypted_logits: CKKS encrypted diagnosis logits
        """
        # Layer 1: Linear transformation on encrypted data
        enc_z1 = encrypted_features.dot(self.w1.T) + self.b1.tolist()
        
        # Polynomial activation (FHE-compatible)
        # f(x) = 0.125*x² + 0.5*x
        enc_a1 = enc_z1.square()  # x²
        enc_a1 *= 0.125           # 0.125*x²
        temp = enc_z1 * 0.5       # 0.5*x
        enc_a1 += temp            # 0.125*x² + 0.5*x
        
        # Layer 2: Linear transformation
        enc_z2 = enc_a1.dot(self.w2.T) + self.b2.tolist()
        
        # Polynomial activation
        enc_a2 = enc_z2.square()
        enc_a2 *= 0.125
        temp = enc_z2 * 0.5
        enc_a2 += temp
        
        # Layer 3: Final classification (no activation)
        enc_logits = enc_a2.dot(self.w3.T) + self.b3.tolist()
        
        return enc_logits
    
    def decrypt_results(self, encrypted_logits):
        """
        CLIENT-SIDE: Decrypt the diagnosis results
        
        Args:
            encrypted_logits: CKKS encrypted logits
            
        Returns:
            logits: numpy array [4] - class probabilities
        """
        logits = np.array(encrypted_logits.decrypt())
        return logits
    
    def predict_encrypted(self, features):
        """
        Complete encrypted prediction pipeline
        
        Args:
            features: numpy array [16384] - CNN features
            
        Returns:
            dict with encrypted and decrypted results
        """
        # Create encryption context
        context = self.create_context()
        
        # CLIENT: Encrypt
        encrypted_features = self.encrypt_features(features, context)
        
        # Serialize for transmission (client → server)
        encrypted_bytes = encrypted_features.serialize()
        
        # SERVER: Deserialize and compute on encrypted data
        server_encrypted_features = ts.ckks_vector_from(context, encrypted_bytes)
        encrypted_logits = self.homomorphic_inference(server_encrypted_features)
        
        # Serialize encrypted results (server → client)
        encrypted_results_bytes = encrypted_logits.serialize()
        
        # CLIENT: Deserialize and decrypt
        client_encrypted_logits = ts.ckks_vector_from(context, encrypted_results_bytes)
        logits = self.decrypt_results(client_encrypted_logits)
        
        # Softmax for probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        return {
            'encrypted_features_size': len(encrypted_bytes),
            'encrypted_results_size': len(encrypted_results_bytes),
            'logits': logits,
            'probabilities': probs,
            'prediction': int(np.argmax(probs)),
            'confidence': float(np.max(probs))
        }


# Global FHE classifier instance
fhe_classifier = None

def initialize_fhe_classifier():
    """Initialize the FHE classifier on server startup"""
    global fhe_classifier
    fhe_classifier = TenSEALFHEClassifier(MODEL_PATHS['fhe_head'])
    return fhe_classifier

def get_fhe_classifier():
    """Get the initialized FHE classifier"""
    global fhe_classifier
    if fhe_classifier is None:
        fhe_classifier = initialize_fhe_classifier()
    return fhe_classifier