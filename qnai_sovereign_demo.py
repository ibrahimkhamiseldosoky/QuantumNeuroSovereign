# ==============================================================================
# IMPORTS
# ==============================================================================
import hashlib
import json
import time
import datetime
import copy
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pennylane for Quantum Circuits
try:
    import pennylane as qml
    from pennylane import numpy as pnp # Use PennyLane's wrapped numpy for parameters
except ImportError:
    print("PennyLane not found. Please install it: pip install pennylane")
    qml = None
    pnp = np 

# PyTorch for Neural Networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch not found. Please install it: pip install torch")
    torch = None
    nn = None
    optim = None

# Optional: Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Matplotlib not found. Plots will be disabled. Install with: pip install matplotlib")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# --- General Config ---
SEED = 42
np.random.seed(SEED)
if torch:
    torch.manual_seed(SEED)

# --- Module 1: Quantum Encoding & Encryption ---
NUM_QUBITS = 4
PQC_LAYERS = 2
LATTICE_SIM_SECURITY_PARAM = 10 
LATTICE_SIM_MODULUS = 12289 # Prime modulus for lattice crypto simulation

# --- Module 2: Brainprint AI ---
BRAINPRINT_DIM = 8
NN_HIDDEN_DIM = 64
NN_EPOCHS = 200 # Increased epochs
NN_LEARNING_RATE = 0.001

# --- Module 4: Quantum Noise ---
NOISE_LEVEL_DEPOLARIZING = 0.01 
NOISE_LEVEL_AMPLITUDE_DAMPING = 0.01
ERROR_MITIGATION_SHOTS = 1000 

# --- Module 6: Multi-Modal ---
IMAGE_FEATURE_DIM = NUM_QUBITS // 2 
TEXT_FEATURE_DIM = NUM_QUBITS - IMAGE_FEATURE_DIM 

# --- Module 8: Blockchain ---
BLOCKCHAIN_DIFFICULTY = 2 

# --- Sample Data ---
SAMPLE_TEXTS = [
    "Quantum computing is the future of computation.",
    "Neural networks can learn complex patterns from data.",
    "Blockchain technology provides decentralized trust and security.",
    "Sovereign AI aims to respect user privacy and control.",
    "Post-quantum cryptography is essential for future security.",
    "Explainable AI helps understand model decisions.",
]
DUMMY_IMAGE_RAW = np.random.rand(4, 4) 

# Global explainer and metrics tracker instances
EXPLAINER = None
METRICS_TRACKER = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def normalize_vector(vec, to_range=(0, np.pi)):
    vec_np = np.asarray(vec, dtype=float) # Ensure float for calculations
    min_val = np.min(vec_np)
    max_val = np.max(vec_np)
    if np.isclose(max_val, min_val): # Avoid division by zero
        return np.full_like(vec_np, (to_range[0] + to_range[1]) / 2.0, dtype=float)
    norm_vec = (vec_np - min_val) / (max_val - min_val + 1e-9) 
    return norm_vec * (to_range[1] - to_range[0]) + to_range[0]

def get_text_embedder(texts, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectorizer.fit(texts)
    return vectorizer

def get_image_embedding(image_raw, feature_dim):
    flat_image = image_raw.flatten()
    if len(flat_image) > feature_dim:
        pool_size = int(np.ceil(len(flat_image) / feature_dim))
        pooled_features = [np.mean(flat_image[i:i+pool_size]) for i in range(0, len(flat_image), pool_size)]
        img_embedding = np.array(pooled_features[:feature_dim])
    elif len(flat_image) < feature_dim:
        img_embedding = np.pad(flat_image, (0, feature_dim - len(flat_image)), 'constant')
    else:
        img_embedding = flat_image
    return normalize_vector(img_embedding, to_range=(0,1))

# ==============================================================================
# MODULE 1: Quantum-Enhanced Thought Encoding and Encryption
# ==============================================================================
class QuantumEncoderDecoder:
    def __init__(self, num_qubits, num_layers=1, explainer=None):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.explainer = explainer
        if not qml:
            self.dev = None
            self.pqc_circuit = None
            self.pqc_weights = None
            if explainer: explainer.add_explanation("QuantumEncoderDecoder", "PennyLane not available. Quantum encoding disabled.")
            return

        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)
        
        @qml.qnode(self.dev, interface="autograd")
        def pqc_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        self.pqc_circuit = pqc_circuit
        self.pqc_weights = pnp.random.uniform(low=0, high=2 * np.pi, 
                                             size=qml.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits),
                                             requires_grad=True)

    def encode(self, classical_embedding):
        if not qml or not self.pqc_circuit:
            if self.explainer: self.explainer.add_explanation("Quantum Encoding", "Skipped: PennyLane not available or PQC not initialized.")
            return np.zeros(self.num_qubits) 

        current_embedding_len = len(classical_embedding)
        if current_embedding_len != self.num_qubits:
            if current_embedding_len > self.num_qubits:
                processed_embedding = classical_embedding[:self.num_qubits]
                if self.explainer: self.explainer.add_explanation("Quantum Encoding", f"Input embedding truncated from {current_embedding_len} to {self.num_qubits} features.")
            else:
                processed_embedding = np.pad(classical_embedding, (0, self.num_qubits - current_embedding_len), 'constant')
                if self.explainer: self.explainer.add_explanation("Quantum Encoding", f"Input embedding padded from {current_embedding_len} to {self.num_qubits} features.")
        else:
            processed_embedding = classical_embedding

        normalized_embedding = normalize_vector(processed_embedding)
        quantum_features = self.pqc_circuit(normalized_embedding, self.pqc_weights)
        if self.explainer: self.explainer.add_explanation("Quantum Encoding", f"Classical embedding encoded into {self.num_qubits}-dim quantum feature vector.")
        return np.array(quantum_features)

    def decode(self, quantum_features_vector):
        if self.explainer: self.explainer.add_explanation("Quantum Decoding", "Quantum feature vector received for 'decoding'.")
        return np.array(quantum_features_vector)

class LatticeCryptoSimulator:
    def __init__(self, security_param=LATTICE_SIM_SECURITY_PARAM, modulus=LATTICE_SIM_MODULUS, explainer=None):
        self.n = security_param 
        self.m = security_param * 2 
        self.q = modulus 
        self.explainer = explainer
        self.pk = None
        self.sk = None

    def keygen(self):
        s_sk = np.random.randint(0, self.q, size=(self.n, 1))    
        A_pk = np.random.randint(0, self.q, size=(self.m, self.n)) 
        e_pk = np.random.normal(0, 2, size=(self.m, 1)).astype(int) 
        P_pk = (A_pk @ s_sk + e_pk) % self.q
        
        self.pk = {'A': A_pk, 'P': P_pk}
        self.sk = {'s': s_sk}
        if self.explainer: self.explainer.add_explanation("Lattice Crypto KeyGen", f"Generated PK (A shape {A_pk.shape}) and SK (s shape {s_sk.shape}).")
        return self.pk, self.sk

    def encrypt(self, public_key, message_vector):
        if public_key is None:
            if self.explainer: self.explainer.add_explanation("Lattice Encrypt Error", "Public key not available.")
            return []
        message_vector_flat = np.asarray(message_vector).flatten()

        A = public_key['A'] 
        P = public_key['P'] 
        
        ciphertexts = []
        m_internal_max = 16 
        message_scaling_factor = (self.q // (2 * (m_internal_max + 1))) 
        if message_scaling_factor == 0: message_scaling_factor = 1 # Avoid zero scaling

        for val_float in message_vector_flat:
            m = int(np.round((val_float + 1.0)/2.0 * m_internal_max)) 
            m = np.clip(m, 0, m_internal_max)
            r_enc = np.random.randint(-1, 2, size=(self.m, 1)) # Small random vector e.g. {-1, 0, 1}
            u = (A.T @ r_enc) % self.q  
            v_scalar_part = (P.T @ r_enc)[0,0] % self.q 
            scaled_message_val = m * message_scaling_factor
            v = (v_scalar_part + scaled_message_val) % self.q
            ciphertexts.append({'u': u.tolist(), 'v': int(v)}) 

        if self.explainer: self.explainer.add_explanation("Lattice Crypto Encrypt", f"Encrypted {len(message_vector_flat)} features into {len(ciphertexts)} LWE ciphertexts.")
        return ciphertexts

    def decrypt(self, private_key, ciphertexts_list):
        if private_key is None:
            if self.explainer: self.explainer.add_explanation("Lattice Decrypt Error", "Private key not available.")
            return np.array([])
        s = private_key['s'] 
        decrypted_values = []
        m_internal_max = 16 
        message_scaling_factor = (self.q // (2 * (m_internal_max + 1)))
        if message_scaling_factor == 0: message_scaling_factor = 1

        for ct in ciphertexts_list:
            u_ct = np.array(ct['u']) 
            v_ct = ct['v']         
            val_approx = (v_ct - (s.T @ u_ct)[0,0]) % self.q
            if val_approx > self.q / 2:
                val_approx -= self.q
            m_recovered = np.round(val_approx / message_scaling_factor)
            m_recovered = int(np.clip(m_recovered, 0, m_internal_max))
            val_unscaled = (m_recovered / m_internal_max) * 2.0 - 1.0
            decrypted_values.append(val_unscaled)
        
        if self.explainer: self.explainer.add_explanation("Lattice Crypto Decrypt", f"Decrypted {len(ciphertexts_list)} ciphertexts to {len(decrypted_values)} features.")
        return np.array(decrypted_values)

# ==============================================================================
# MODULE 2: Personalized AI Thought Decryptor ("Brainprint")
# ==============================================================================
if nn:
    class BrainprintDecryptorNN(nn.Module):
        def __init__(self, input_dim, brainprint_dim, output_dim, hidden_dim=NN_HIDDEN_DIM):
            super().__init__()
            self.fc1 = nn.Linear(input_dim + brainprint_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.tanh_out = nn.Tanh() # Output is in [-1, 1] like PQC expvals
            self.input_dim = input_dim 

        def forward(self, aggregated_ciphertext_features, brainprint_embedding):
            if not isinstance(aggregated_ciphertext_features, torch.Tensor):
                aggregated_ciphertext_features = torch.tensor(aggregated_ciphertext_features, dtype=torch.float32)
            if not isinstance(brainprint_embedding, torch.Tensor):
                brainprint_embedding = torch.tensor(brainprint_embedding, dtype=torch.float32)

            if aggregated_ciphertext_features.ndim == 1:
                aggregated_ciphertext_features = aggregated_ciphertext_features.unsqueeze(0)
            if brainprint_embedding.ndim == 1:
                brainprint_embedding = brainprint_embedding.unsqueeze(0)
            
            x = torch.cat((aggregated_ciphertext_features, brainprint_embedding), dim=1)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x) 
            x = self.tanh_out(x) # Constrain output
            return x

    def aggregate_ciphertext_features(ciphertexts_list, expected_dim):
        if not ciphertexts_list:
            return np.zeros(expected_dim)
        
        all_u_vectors = [np.array(ct['u']).flatten() for ct in ciphertexts_list if 'u' in ct and len(ct['u']) == LATTICE_SIM_SECURITY_PARAM]
        all_v_values = [ct['v'] for ct in ciphertexts_list if 'v' in ct]

        if all_u_vectors:
            mean_u = np.mean(all_u_vectors, axis=0)
        else:
            mean_u = np.zeros(LATTICE_SIM_SECURITY_PARAM) 
        
        mean_v = np.mean(all_v_values) if all_v_values else 0
        agg_features_raw = np.concatenate((mean_u, [mean_v]))
        
        # Pad or truncate
        if len(agg_features_raw) > expected_dim:
            agg_features = agg_features_raw[:expected_dim]
        elif len(agg_features_raw) < expected_dim:
            agg_features = np.pad(agg_features_raw, (0, expected_dim - len(agg_features_raw)), 'constant')
        else:
            agg_features = agg_features_raw
        return agg_features


    def _derive_lattice_sk_from_brainprint(brainprint, n_dim, q_mod):
        bp_bytes = brainprint.astype(np.float32).tobytes()
        h_digest = hashlib.sha256(bp_bytes).digest()
        sk_vec = []
        bytes_per_element = 2 
        num_hash_bytes_needed = n_dim * bytes_per_element
        
        if len(h_digest) < num_hash_bytes_needed:
            h_digest += b'\x00' * (num_hash_bytes_needed - len(h_digest))

        for i in range(n_dim):
            val = int.from_bytes(h_digest[i*bytes_per_element:(i+1)*bytes_per_element], 'big', signed=False) 
            sk_vec.append(val % q_mod)
        return np.array(sk_vec).reshape(n_dim, 1)

    def train_brainprint_decryptor(model, lattice_crypto, pqc_encoder, sample_embeddings, sample_brainprints, epochs, lr, explainer, metrics_tracker):
        if not nn or not pqc_encoder.pqc_circuit: # Need PQC for training data
            if explainer: explainer.add_explanation("Brainprint NN Training", "Skipped: PyTorch or PQC not available.")
            return []
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        A_shared = np.random.randint(0, lattice_crypto.q, size=(lattice_crypto.m, lattice_crypto.n))

        training_losses = []
        model.train() 
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for i in range(len(sample_embeddings)):
                plaintext_embedding = sample_embeddings[i]
                correct_brainprint = sample_brainprints[i]

                q_embedding = pqc_encoder.encode(plaintext_embedding)
                if q_embedding is None or np.all(q_embedding == 0): # Skip if PQC failed
                    continue

                current_sk_s = _derive_lattice_sk_from_brainprint(correct_brainprint, lattice_crypto.n, lattice_crypto.q)
                e_pk_train = np.random.normal(0, 2, size=(lattice_crypto.m, 1)).astype(int)
                current_P_pk = (A_shared @ current_sk_s + e_pk_train) % lattice_crypto.q
                current_pk_for_nn = {'A': A_shared, 'P': current_P_pk}
                
                ciphertexts = lattice_crypto.encrypt(current_pk_for_nn, q_embedding)
                if not ciphertexts: continue # Skip if encryption failed
                
                agg_ct_feat_raw = aggregate_ciphertext_features(ciphertexts, model.input_dim)
                agg_ct_feat_normalized = agg_ct_feat_raw / LATTICE_SIM_MODULUS # Normalize

                target_tensor = torch.tensor(q_embedding, dtype=torch.float32).unsqueeze(0) 
                
                optimizer.zero_grad()
                predicted_q_embedding = model(agg_ct_feat_normalized, correct_brainprint)
                loss = criterion(predicted_q_embedding, target_tensor)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                training_losses.append(avg_epoch_loss)
                if (epoch + 1) % 20 == 0 or epoch == epochs -1 : # Print less frequently for more epochs
                    print(f"Brainprint Decryptor Training: Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
            else:
                print(f"Brainprint Decryptor Training: Epoch {epoch+1}/{epochs}, No valid batches processed.")
                training_losses.append(float('inf')) # Indicate problem
        
        if explainer and training_losses: explainer.add_explanation("Brainprint NN Training", f"Completed {epochs} epochs. Final loss: {training_losses[-1]:.6f}")
        if metrics_tracker and training_losses: metrics_tracker.log_metric("Brainprint NN Final Loss", training_losses[-1])
        return training_losses

# ==============================================================================
# MODULE 3: Zero-Knowledge-Like Proof of Mind Matching
# ==============================================================================
class ZKPLikeProof:
    def __init__(self, explainer=None):
        self.explainer = explainer
        self.p_zkp = 1000000007 # Prime modulus for Z_p field operations
        self.order_q_zkp = self.p_zkp - 1 # Order of the multiplicative group Z_p^*
        self.g_zkp = 5 # Primitive root mod 1000000007

    def _hash_inputs_to_scalar(self, *args):
        hasher = hashlib.sha256()
        for arg in args:
            hasher.update(str(arg).encode())
        s = (int(hasher.hexdigest(), 16) % (self.order_q_zkp - 1)) + 1 # Map to [1, q-1]
        return s

    def prove_knowledge(self, secret_brainprint_embedding): # Prover
        s_scalar = self._hash_inputs_to_scalar(secret_brainprint_embedding.tobytes())
        v_public_id = pow(self.g_zkp, s_scalar, self.p_zkp) # v = g^s mod p
        if self.explainer: self.explainer.add_explanation("ZKP Prover", f"Secret s={s_scalar}. Public ID v={v_public_id}.")

        r_blinding = np.random.randint(1, self.order_q_zkp) # r in [1, q-1]
        t_commitment = pow(self.g_zkp, r_blinding, self.p_zkp) # t = g^r mod p
        if self.explainer: self.explainer.add_explanation("ZKP Prover", f"Random r={r_blinding}, commitment t={t_commitment}.")
        return s_scalar, v_public_id, r_blinding, t_commitment

    def verifier_generates_challenge(self): # Verifier
        c_challenge = np.random.randint(1, self.p_zkp) # Challenge c can be < p
        if self.explainer: self.explainer.add_explanation("ZKP Verifier", f"Generated challenge c={c_challenge}.")
        return c_challenge

    def prover_generates_response(self, s_scalar, r_blinding, c_challenge): # Prover
        z_response = (r_blinding + c_challenge * s_scalar) % self.order_q_zkp 
        if self.explainer: self.explainer.add_explanation("ZKP Prover", f"Calculated response z={z_response}.")
        return z_response

    def verify_proof(self, v_public_id, t_commitment, c_challenge, z_response): # Verifier
        lhs = pow(self.g_zkp, z_response, self.p_zkp)
        v_pow_c = pow(v_public_id, c_challenge, self.p_zkp)
        rhs = (t_commitment * v_pow_c) % self.p_zkp
        
        verification_passed = (lhs == rhs)
        if self.explainer: self.explainer.add_explanation("ZKP Verifier", f"Check g^z ({lhs}) vs t*v^c ({rhs}). Verification: {verification_passed}.")
        return verification_passed

# ==============================================================================
# MODULE 4: Quantum Noise-Resilience Simulation
# ==============================================================================
class QuantumNoiseSimulator:
    def __init__(self, num_qubits, explainer=None, metrics_tracker=None):
        self.num_qubits = num_qubits
        self.explainer = explainer
        self.metrics_tracker = metrics_tracker

    def _apply_noise_layer(self, noise_level_dp, noise_level_ad):
        for i in range(self.num_qubits):
            if noise_level_dp > 0: qml.DepolarizingChannel(noise_level_dp, wires=i)
            if noise_level_ad > 0: qml.AmplitudeDamping(noise_level_ad, wires=i)
    
    def simulate_noise_impact(self, ideal_encoder_decoder, classical_embedding, noise_level_dp, noise_level_ad, shots_for_mitigation):
        if not qml or not ideal_encoder_decoder.pqc_circuit:
            if self.explainer: self.explainer.add_explanation("Noise Simulation", "Skipped: PennyLane or PQC not available.")
            return None, None, None

        processed_embedding = np.asarray(classical_embedding) 
        if len(processed_embedding) != self.num_qubits:
             processed_embedding = processed_embedding[:self.num_qubits] if len(processed_embedding) > self.num_qubits else np.pad(processed_embedding, (0, self.num_qubits - len(processed_embedding)), 'constant')
        normalized_embedding = normalize_vector(processed_embedding)
        pqc_weights = ideal_encoder_decoder.pqc_weights

        ideal_q_features = ideal_encoder_decoder.encode(classical_embedding) 

        noisy_dev_expval = qml.device("default.mixed", wires=self.num_qubits, shots=shots_for_mitigation)
        @qml.qnode(noisy_dev_expval, interface="autograd")
        def noisy_expval_qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            self._apply_noise_layer(noise_level_dp, noise_level_ad)
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            self._apply_noise_layer(noise_level_dp, noise_level_ad)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        noisy_q_features_raw = noisy_expval_qnode(normalized_embedding, pqc_weights)
        
        ideal_dev_statevec = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(ideal_dev_statevec, interface="autograd")
        def ideal_state_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.state()
        ideal_state_vector = ideal_state_circuit(normalized_embedding, pqc_weights)

        noisy_dev_densitymatrix = qml.device("default.mixed", wires=self.num_qubits)
        @qml.qnode(noisy_dev_densitymatrix, interface="autograd")
        def noisy_density_matrix_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            self._apply_noise_layer(noise_level_dp, noise_level_ad)
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            self._apply_noise_layer(noise_level_dp, noise_level_ad)
            return qml.state()
        noisy_density_matrix = noisy_density_matrix_circuit(normalized_embedding, pqc_weights)
        
        state_fidelity = 0.0
        try:
            state_fidelity = qml.math.fidelity(ideal_state_vector, noisy_density_matrix)
        except Exception as e:
            ideal_state_vector_c = ideal_state_vector.astype(np.complex128)
            try:
                state_fidelity = np.real(np.vdot(ideal_state_vector_c, noisy_density_matrix @ ideal_state_vector_c))
            except ValueError as ve: 
                 if self.explainer: self.explainer.add_explanation("Noise Sim Error", f"Fidelity calc error (shape): {ve}")
                 state_fidelity = -1 
            if self.explainer: self.explainer.add_explanation("Noise Sim Warning", f"Fidelity calculation fallback used due to: {e}")

        if self.explainer:
            self.explainer.add_explanation("Noise Simulation", f"Ideal q_features: {np.round(ideal_q_features,3)}")
            self.explainer.add_explanation("Noise Simulation", f"Noisy q_features (DP={noise_level_dp}, AD={noise_level_ad}, Shots={shots_for_mitigation}): {np.round(noisy_q_features_raw,3)}")
            self.explainer.add_explanation("Noise Simulation", f"State fidelity (ideal pure vs noisy mixed): {state_fidelity:.4f}")
        if self.metrics_tracker:
            self.metrics_tracker.log_metric("Quantum State Fidelity (Noisy)", state_fidelity)
        return ideal_q_features, np.array(noisy_q_features_raw), state_fidelity

# ==============================================================================
# MODULE 5: Adaptive Quantum Circuit Architecture Optimization
# ==============================================================================
class PQCAutomatedOptimizer:
    def __init__(self, explainer=None, metrics_tracker=None):
        self.explainer = explainer
        self.metrics_tracker = metrics_tracker

    def optimize_pqc_depth(self, sample_classical_embedding, num_qubits, potential_depths=[1, 2, 3]):
        if not qml:
            if self.explainer: self.explainer.add_explanation("PQC Optimization", "Skipped: PennyLane not available.")
            return PQC_LAYERS 
            
        best_depth = PQC_LAYERS
        best_metric = -1.0
        if self.explainer: self.explainer.add_explanation("PQC Optimization", f"Starting PQC depth optimization. Trying depths: {potential_depths}.")

        processed_embedding = np.asarray(sample_classical_embedding)
        if len(processed_embedding) != num_qubits:
            processed_embedding = processed_embedding[:num_qubits] if len(processed_embedding) > num_qubits else np.pad(processed_embedding, (0, num_qubits - len(processed_embedding)),'constant')
        normalized_input = normalize_vector(processed_embedding)

        for depth in potential_depths:
            dev_opt = qml.device("default.qubit", wires=num_qubits)
            @qml.qnode(dev_opt, interface="autograd")
            def temp_pqc(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
            
            temp_weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=num_qubits)
            temp_weights = pnp.random.uniform(low=0, high=2*np.pi, size=temp_weights_shape)
            
            q_features = temp_pqc(normalized_input, temp_weights)
            current_metric = np.var(q_features) 
            
            if self.explainer: self.explainer.add_explanation("PQC Optimization", f"Depth {depth}: q_features variance = {current_metric:.4f}")
            if current_metric > best_metric:
                best_metric = current_metric
                best_depth = depth
        
        if self.explainer: self.explainer.add_explanation("PQC Optimization", f"Optimal PQC depth found: {best_depth} with metric (variance) {best_metric:.4f}.")
        if self.metrics_tracker:
            self.metrics_tracker.log_metric("Optimized PQC Depth", best_depth)
            self.metrics_tracker.log_metric("PQC Opt Metric (Variance)", best_metric)
        return best_depth

# ==============================================================================
# MODULE 6: Multi-Modal Thought Fusion
# ==============================================================================
class MultiModalEncoder:
    def __init__(self, text_embedder, text_feature_dim, image_feature_dim, explainer=None):
        self.text_embedder = text_embedder
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.total_features = text_feature_dim + image_feature_dim
        self.explainer = explainer
        if self.explainer: self.explainer.add_explanation("MultiModal Encoder Init", f"Initialized for text ({text_feature_dim}D) and image ({image_feature_dim}D). Total: {self.total_features}D.")

    def get_fused_embedding(self, text_input, image_input_raw):
        text_vector_sparse = self.text_embedder.transform([text_input])
        text_vector_dense_raw = text_vector_sparse.toarray().flatten()
        
        if len(text_vector_dense_raw) < self.text_feature_dim:
            text_vector_dense = np.pad(text_vector_dense_raw, (0, self.text_feature_dim - len(text_vector_dense_raw)), 'constant')
        elif len(text_vector_dense_raw) > self.text_feature_dim:
            text_vector_dense = text_vector_dense_raw[:self.text_feature_dim]
        else: text_vector_dense = text_vector_dense_raw

        image_embedding_vec = get_image_embedding(image_input_raw, self.image_feature_dim)
        fused_embedding = np.concatenate((text_vector_dense, image_embedding_vec))
        
        if self.explainer: self.explainer.add_explanation("MultiModal Fusion", f"Fused text ({len(text_vector_dense)}D) and image ({len(image_embedding_vec)}D) into {len(fused_embedding)}D embedding.")
        return fused_embedding

# ==============================================================================
# MODULE 7: Explainability Module
# ==============================================================================
class Explainer:
    def __init__(self): self.explanations = []
    def add_explanation(self, step_name, details):
        explanation = f"[{step_name}]: {details}"
        self.explanations.append(explanation)
        print(f"LOG: {explanation}") 
    def get_summary_report(self):
        report = "Explainability Summary Report:\n" + "="*30 + "\n"
        if not self.explanations: report += "No explanations recorded.\n"
        for i, expl in enumerate(self.explanations): report += f"{i+1}. {expl}\n"
        report += "="*30 + "\n"
        return report
    def clear(self): self.explanations = []

# ==============================================================================
# MODULE 8: Blockchain-Based Thought Ledger (Local Simulation)
# ==============================================================================
class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data 
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({"index": self.index, "timestamp": str(self.timestamp), 
                                   "data": self.data, "previous_hash": self.previous_hash, 
                                   "nonce": self.nonce}, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self, difficulty=BLOCKCHAIN_DIFFICULTY, explainer=None):
        self.chain = [self._create_genesis_block()]
        self.difficulty = difficulty 
        self.explainer = explainer
        if self.explainer: self.explainer.add_explanation("Blockchain Init", f"Blockchain initialized with PoW difficulty {difficulty}.")

    def _create_genesis_block(self): return Block(0, datetime.datetime.now(), "Genesis Block", "0")
    def get_latest_block(self): return self.chain[-1]

    def add_block(self, data_hashed_encrypted_thought):
        new_block = Block(len(self.chain), datetime.datetime.now(), data_hashed_encrypted_thought, self.get_latest_block().hash)
        self._proof_of_work(new_block) 
        self.chain.append(new_block)
        if self.explainer: self.explainer.add_explanation("Blockchain Add Block", f"Block #{new_block.index} added (Nonce: {new_block.nonce}).")
        return new_block

    def _proof_of_work(self, block):
        target_prefix = '0' * self.difficulty
        while not block.hash.startswith(target_prefix):
            block.nonce += 1
            block.hash = block.calculate_hash()
        if self.explainer: self.explainer.add_explanation("Blockchain PoW", f"PoW found for block #{block.index} (Nonce: {block.nonce}).")

    def is_chain_valid(self):
        target_prefix = '0' * self.difficulty
        for i in range(1, len(self.chain)):
            current_block, previous_block = self.chain[i], self.chain[i-1]
            if current_block.hash != current_block.calculate_hash(): return False
            if current_block.previous_hash != previous_block.hash: return False
            if not current_block.hash.startswith(target_prefix): return False
        if self.explainer: self.explainer.add_explanation("Blockchain Validation", "Integrity: VALID.")
        return True

    def print_chain(self):
        print("\n--- Thought Ledger (Blockchain) ---")
        for block in self.chain:
            print(f"Block #{block.index} | Hash: {block.hash[:10]}... | PrevHash: {block.previous_hash[:10]}... | Nonce: {block.nonce} | Data: {str(block.data)[:30]}...")
        print("--- End of Ledger ---")

# ==============================================================================
# MODULE 9: Performance Metrics & Visualization
# ==============================================================================
class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.plots_dir = "plots"
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def start_timer(self, metric_name): self.start_times[metric_name] = time.time()
    def stop_timer(self, metric_name):
        if metric_name in self.start_times:
            elapsed_time = time.time() - self.start_times[metric_name]
            self.log_metric(f"{metric_name} Time (s)", elapsed_time)
            del self.start_times[metric_name]
            return elapsed_time
        return None
    def log_metric(self, name, value):
        self.metrics[name] = value
        print(f"METRIC: {name} = {value:.4f}" if isinstance(value, (float, np.float_)) else f"METRIC: {name} = {value}")

    def display_summary(self):
        print("\n--- Performance Metrics Summary ---")
        if not self.metrics: print("No metrics recorded.")
        else:
            for name, value in self.metrics.items():
                val_str = f"{float(value):.4f}" if isinstance(value, (float, np.float_, np.float32, np.float64)) else str(value)
                print(f"  {name}: {val_str}")
        print("--- End of Metrics ---")
    
    def _save_plot(self, title, filename_suffix=""):
        plot_filename = title.replace(" ", "_").replace("/", "_").lower() + filename_suffix + ".png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        try:
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving plot {plot_path}: {e}")
        plt.close()


    def plot_bar_chart(self, values, labels, title="Bar Chart", ylabel="Value", filename_suffix=""):
        if not plt or not values or not labels:
            print(f"Plotting '{title}' skipped: Matplotlib unavailable or no data.")
            if values and labels: print("Data:", list(zip(labels, values)))
            return

        plt.figure(figsize=(max(8, len(labels)*0.8), 6))
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'lightsalmon', 'cyan']
        bar_colors = [colors[i % len(colors)] for i in range(len(values))]
        plt.bar(labels, values, color=bar_colors)
        plt.ylabel(ylabel)
        plt.title(title)
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        # Adjust y-limits to give some padding, ensure 0 is included if values are positive/negative
        y_bottom = min(0, min_val - 0.1 * abs(min_val)) if min_val < 0 else 0
        y_top = max(0.1, max_val + 0.1 * abs(max_val)) if max_val > 0 else 0.1 # Ensure some positive space if max_val is 0 or negative
        if y_bottom >= y_top : y_top = y_bottom + 0.1 # Ensure top is greater than bottom

        plt.ylim(y_bottom, y_top)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        self._save_plot(title, filename_suffix)


    def plot_line_data(self, y_data, x_data=None, title="Line Plot", xlabel="X-axis", ylabel="Y-axis", filename_suffix=""):
        if not plt or not y_data :
            print(f"Plotting '{title}' skipped: Matplotlib unavailable or no y_data.")
            if y_data: print(f"Y_Data for {title}: {y_data[:5]}...")
            return
        
        plt.figure(figsize=(8, 5))
        if x_data is not None and len(x_data) == len(y_data):
            plt.plot(x_data, y_data, marker='o', linestyle='-')
        else:
            plt.plot(y_data, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        self._save_plot(title, filename_suffix)

# ==============================================================================
# MODULE 10: Security Stress Test
# ==============================================================================
def run_security_stress_test(brainprint_nn_model, original_q_embedding, ciphertext_agg_features, correct_brainprint, 
                             num_wrong_attempts=10, explainer=None, metrics_tracker=None):
    if not nn or not brainprint_nn_model:
        if explainer: explainer.add_explanation("Security Stress Test", "Skipped: PyTorch or model not available.")
        return None, [] 

    if explainer: explainer.add_explanation("Security Stress Test", f"Starting stress test with {num_wrong_attempts} wrong brainprints.")
    
    brainprint_nn_model.eval() 
    with torch.no_grad():
        # Normalize aggregated ciphertext features for NN input
        agg_ct_tensor_norm = torch.tensor(ciphertext_agg_features / LATTICE_SIM_MODULUS, dtype=torch.float32) 
        correct_bp_tensor = torch.tensor(correct_brainprint, dtype=torch.float32)
        if agg_ct_tensor_norm.ndim == 1: agg_ct_tensor_norm = agg_ct_tensor_norm.unsqueeze(0)
        if correct_bp_tensor.ndim == 1: correct_bp_tensor = correct_bp_tensor.unsqueeze(0)
            
        correct_pred_q_emb_tensor = brainprint_nn_model(agg_ct_tensor_norm, correct_bp_tensor)
        correct_pred_q_emb = correct_pred_q_emb_tensor.cpu().numpy().flatten()
    
    similarity_threshold = 0.80 
    correct_cos_sim = cosine_similarity(original_q_embedding.reshape(1,-1), correct_pred_q_emb.reshape(1,-1))[0,0]
    
    if explainer: explainer.add_explanation("Security Stress Test", f"Decryption with correct brainprint: Cosine Sim = {correct_cos_sim:.4f}")
    if metrics_tracker: metrics_tracker.log_metric("Security Test - Correct BP Sim", correct_cos_sim)

    successful_wrong_decryptions = 0
    all_wrong_similarities = []
    for i in range(num_wrong_attempts):
        wrong_brainprint = np.random.rand(BRAINPRINT_DIM).astype(np.float32) 
        while np.allclose(wrong_brainprint, correct_brainprint): 
            wrong_brainprint = np.random.rand(BRAINPRINT_DIM).astype(np.float32)
        
        with torch.no_grad():
            wrong_bp_tensor = torch.tensor(wrong_brainprint, dtype=torch.float32)
            if wrong_bp_tensor.ndim == 1: wrong_bp_tensor = wrong_bp_tensor.unsqueeze(0)
            # Use the same normalized agg_ct_tensor_norm
            wrong_pred_q_emb_tensor = brainprint_nn_model(agg_ct_tensor_norm, wrong_bp_tensor) 
            wrong_pred_q_emb = wrong_pred_q_emb_tensor.cpu().numpy().flatten()
        
        wrong_cos_sim = cosine_similarity(original_q_embedding.reshape(1,-1), wrong_pred_q_emb.reshape(1,-1))[0,0]
        all_wrong_similarities.append(wrong_cos_sim)
        if explainer: explainer.add_explanation("Security Stress Test", f"Attempt {i+1} with wrong brainprint: Cosine Sim = {wrong_cos_sim:.4f}")
        if wrong_cos_sim > similarity_threshold:
            successful_wrong_decryptions += 1
            if explainer: explainer.add_explanation("Security Stress Test", f"Potential breach! Wrong BP Sim: {wrong_cos_sim:.4f} > threshold {similarity_threshold}.")

    summary_msg = f"Results: {successful_wrong_decryptions}/{num_wrong_attempts} successful wrong decryptions (threshold {similarity_threshold})."
    if explainer: explainer.add_explanation("Security Stress Test", summary_msg)
    print(summary_msg)
    if metrics_tracker:
        metrics_tracker.log_metric("Security Test - Wrong Decryptions Count", successful_wrong_decryptions)
        metrics_tracker.log_metric("Security Test - Total Attempts", num_wrong_attempts)
    return correct_cos_sim, all_wrong_similarities

# ==============================================================================
# MODULE 11: Documentation & Research Proposal Generation
# ==============================================================================
def generate_research_proposal_md(filename="QuantumNeuroAISovereign_Proposal.md"):
    content = f"""
# Research Proposal: Quantum-Neuro AI Sovereign 2.0

## 1. Motivation and Background

The Quantum-Neuro AI Sovereign 2.0 system aims to explore the synergistic potential of quantum computing, neural networks, post-quantum cryptography, and blockchain technologies to create a futuristic model for secure and personalized thought processing. In an era of increasing data ubiquity and concerns over digital sovereignty, this project investigates novel paradigms for encoding, encrypting, and managing sensitive cognitive information. 

Key motivations include:
- **Enhanced Privacy:** Leveraging quantum mechanics and post-quantum cryptography for robust protection of thought data.
- **Personalized Security:** Introducing "brainprints" as unique biometric keys managed by AI, ensuring only the authorized individual can access their thoughts.
- **Verifiable Control:** Using zero-knowledge-like proofs to assert ownership or knowledge without revealing underlying sensitive data.
- **Resilient Systems:** Exploring quantum noise resilience and adaptive architectures for practical future implementations.
- **Decentralized Auditability:** Utilizing blockchain for a tamper-proof ledger of (hashed, encrypted) thought records.

This demo project serves as a conceptual prototype, integrating these advanced concepts into a runnable Python simulation to showcase their combined potential and identify areas for future research.

## 2. Technical Architecture

The system is composed of several interconnected modules:

**2.1. Quantum-Enhanced Thought Encoding and Encryption:**
   - Classical text/multimodal embeddings are transformed by a Parameterized Quantum Circuit (PQC) into "quantum feature vectors."
   - These quantum feature vectors are then encrypted using a simulated lattice-based post-quantum cryptography scheme (LWE-like).
   - **Pipeline:** Text/Image -> Embedding -> PQC (Quantum Encoding) -> Lattice Encryption.
   - **Decryption Path:** Ciphertext -> Lattice Decryption -> Quantum Feature Vector -> Comparison/Reconstruction.

**2.2. Personalized AI Thought Decryptor ("Brainprint"):**
   - A small neural network (PyTorch-based) acts as a personalized decryption mechanism.
   - It requires a "brainprint" embedding (unique user identifier) to successfully process/decrypt ciphertexts.
   - Demonstrates how AI can gate access based on personalized biometric-like keys.

**2.3. Zero-Knowledge-Like Proof of Mind Matching:**
   - A simplified Schnorr-like protocol simulation demonstrates that the system (or user) "knows" the correct brainprint associated with a thought, without revealing the brainprint itself or the thought content.

**2.4. Quantum Noise-Resilience Simulation:**
   - Models depolarizing and amplitude damping noise effects on the PQC.
   - Implements basic mitigation (e.g., statistical averaging from multiple shots).
   - Visualizes (textually or via plot) the impact of noise on quantum state fidelity or output vector accuracy.

**2.5. Adaptive Quantum Circuit Architecture Optimization:**
   - A rudimentary module attempts to adjust PQC parameters (e.g., circuit depth) by evaluating a simple heuristic (e.g., output feature variance) to enhance encoding distinctiveness.

**2.6. Multi-Modal Thought Fusion:**
   - Extends input beyond text to include simple image data.
   - Fuses text and image embeddings into a joint vector for quantum encoding.

**2.7. Explainability Module:**
   - Provides step-by-step textual explanations of the system's operations, enhancing transparency.

**2.8. Blockchain-Based Thought Ledger (Local Simulation):**
   - A lightweight Python blockchain stores hashes of encrypted thoughts with timestamps, secured by simple Proof-of-Work.
   - Demonstrates tamper-resistant logging for auditability.

**2.9. Performance Metrics & Visualization:**
   - Tracks metrics like encryption/decryption time, quantum circuit fidelity/output accuracy, and neural decryption confidence.
   - Outputs summaries and (optionally) basic plots.

**2.10. Security Stress Test:**
   - An automated script attempts to decrypt ciphertexts using multiple incorrect brainprints to evaluate the robustness of the personalized AI decryptor.

## 3. Demo Features Showcased

- **End-to-End Pipeline:** From text/image input to encrypted storage and authorized decryption.
- **Quantum Encoding:** Use of PennyLane for PQC-based feature transformation.
- **Post-Quantum Encryption (Simulated):** Placeholder LWE-like encryption for conceptual demonstration.
- **AI-Gated Decryption:** Neural network requiring a correct "brainprint."
- **ZKP Simulation:** Proof of brainprint knowledge.
- **Noise Effects:** Impact of quantum noise and basic mitigation.
- **Adaptive PQC (Basic):** Simple circuit parameter adjustment.
- **Multi-Modal Input:** Handling text and image data.
- **Step-by-Step Explanations:** Enhanced transparency.
- **Blockchain Ledgering:** Secure logging of thought metadata.
- **Security Evaluation:** Resistance to unauthorized access attempts.

## 4. Future Research Directions

This demo opens avenues for extensive future research:

- **Advanced PQC Schemes:** Implement and evaluate mature post-quantum lattice-based cryptosystems (e.g., Kyber, NTRU) using available libraries or further simulations.
- **Sophisticated Brainprint AI:** Develop more complex neural network architectures for brainprint generation, verification, and continuous authentication. Explore federated learning for privacy-preserving model training.
- **Robust ZKP Protocols:** Integrate more formal and secure ZKP schemes (e.g., zk-SNARKs, Bulletproofs) for mind matching and other attestations.
- **Real Quantum Hardware Integration:** Test PQC components on actual quantum processors via cloud platforms. Develop error correction and mitigation strategies tailored to specific hardware.
- **Advanced Adaptive PQC:** Employ reinforcement learning or Bayesian optimization for dynamic PQC architecture and parameter tuning.
- **Semantic Thought Reconstruction:** Improve decoding from quantum/encrypted states back to richer semantic representations, possibly using generative AI models.
- **Ethical Frameworks:** Develop comprehensive ethical guidelines and governance models for sovereign AI and thought-data technologies.
- **Scalability and Performance:** Optimize all components for real-world performance and scalability.
- **Biometric Brainprints:** Investigate methods for deriving actual brainprints from neuro-signals (EEG, fMRI) and integrating them into the system.
- **User Interface and Experience:** Design intuitive interfaces for users to manage their sovereign thoughts and cryptographic keys.

## 5. Conclusion

The Quantum-Neuro AI Sovereign 2.0 demo provides a foundational glimpse into a future where advanced computational paradigms converge to offer unprecedented levels of security, personalization, and control over digital cognitive information. While conceptual, it highlights key research challenges and opportunities at the intersection of these cutting-edge fields.
"""
    try:
        with open(filename, "w", encoding="utf-8") as f: f.write(content)
        print(f"Research proposal generated: {filename}")
        if EXPLAINER: EXPLAINER.add_explanation("Documentation", f"Proposal MD generated: {filename}.")
    except Exception as e:
        print(f"Error generating research proposal: {e}")
        if EXPLAINER: EXPLAINER.add_explanation("Doc Error", f"Failed proposal MD: {e}.")

# ==============================================================================
# MAIN DEMO SCRIPT / ORCHESTRATION
# ==============================================================================
def main_demo():
    global EXPLAINER, METRICS_TRACKER 
    EXPLAINER = Explainer()
    METRICS_TRACKER = MetricsTracker()

    print("="*50 + "\nInitializing Quantum-Neuro AI Sovereign 2.0 Demo\n" + "="*50)
    EXPLAINER.add_explanation("System Init", "Demo started.")

    effective_text_feature_dim = TEXT_FEATURE_DIM if (TEXT_FEATURE_DIM > 0 and IMAGE_FEATURE_DIM > 0 and TEXT_FEATURE_DIM + IMAGE_FEATURE_DIM == NUM_QUBITS) else NUM_QUBITS
    text_embedder_global = get_text_embedder(SAMPLE_TEXTS, max_features=effective_text_feature_dim)

    # --- MODULE 1: Basic Text Thought Encoding & Encryption ---
    print("\n--- DEMO: Basic Text Thought Processing ---")
    EXPLAINER.add_explanation("Demo Section Start", "Basic Text Processing")
    METRICS_TRACKER.start_timer("M1_Quantum_Encoding")
    q_encoder_decoder = QuantumEncoderDecoder(NUM_QUBITS, PQC_LAYERS, EXPLAINER)
    
    sample_text_input = SAMPLE_TEXTS[0]
    text_emb_sparse = text_embedder_global.transform([sample_text_input])
    text_embedding_original_raw = text_emb_sparse.toarray().flatten()
    
    if len(text_embedding_original_raw) < NUM_QUBITS:
        text_embedding_original = np.pad(text_embedding_original_raw, (0, NUM_QUBITS - len(text_embedding_original_raw)), 'constant')
    elif len(text_embedding_original_raw) > NUM_QUBITS:
         text_embedding_original = text_embedding_original_raw[:NUM_QUBITS]
    else: text_embedding_original = text_embedding_original_raw

    EXPLAINER.add_explanation("Text Embedding", f"Input: '{sample_text_input}'. TF-IDF ({len(text_embedding_original)}D): {np.round(text_embedding_original[:3],3)}...")
    
    q_features_original = q_encoder_decoder.encode(text_embedding_original)
    METRICS_TRACKER.stop_timer("M1_Quantum_Encoding")
    print(f"Original Text Emb (shape {text_embedding_original.shape}): {np.round(text_embedding_original[:3],3)}...")
    print(f"Quantum-Encoded Features (shape {q_features_original.shape if q_features_original is not None else 'N/A'}): {np.round(q_features_original[:3],3) if q_features_original is not None else 'N/A'}...")

    METRICS_TRACKER.start_timer("M1_Lattice_Crypto_KeyGen_Encrypt")
    lattice_crypto = LatticeCryptoSimulator(explainer=EXPLAINER)
    pk, sk = lattice_crypto.keygen()
    encrypted_q_features = lattice_crypto.encrypt(pk, q_features_original)
    METRICS_TRACKER.stop_timer("M1_Lattice_Crypto_KeyGen_Encrypt")
    if encrypted_q_features: print(f"Encrypted QF (first CT): u={np.array(encrypted_q_features[0]['u'])[:2].tolist()}..., v={encrypted_q_features[0]['v']}")
    else: print("Encrypted QF: No data.")

    METRICS_TRACKER.start_timer("M1_Lattice_Crypto_Decrypt")
    decrypted_q_features = lattice_crypto.decrypt(sk, encrypted_q_features)
    METRICS_TRACKER.stop_timer("M1_Lattice_Crypto_Decrypt")
    print(f"Decrypted QF: {np.round(decrypted_q_features[:3],3) if decrypted_q_features is not None and len(decrypted_q_features)>0 else 'N/A'}...")

    best_match_text, max_similarity = "N/A", -1.0
    if q_encoder_decoder.pqc_circuit is not None and decrypted_q_features is not None and len(decrypted_q_features) > 0:
        for text_sample in SAMPLE_TEXTS:
            sample_emb_s = text_embedder_global.transform([text_sample])
            sample_emb_d_raw = sample_emb_s.toarray().flatten()
            if len(sample_emb_d_raw) < NUM_QUBITS: sample_emb_d = np.pad(sample_emb_d_raw, (0, NUM_QUBITS - len(sample_emb_d_raw)), 'constant')
            elif len(sample_emb_d_raw) > NUM_QUBITS: sample_emb_d = sample_emb_d_raw[:NUM_QUBITS]
            else: sample_emb_d = sample_emb_d_raw
            q_features_sample = q_encoder_decoder.encode(sample_emb_d) 
            sim = cosine_similarity(decrypted_q_features.reshape(1,-1), q_features_sample.reshape(1,-1))[0,0]
            if sim > max_similarity: max_similarity, best_match_text = sim, text_sample
    EXPLAINER.add_explanation("Text Reconstruction", f"Best match: '{best_match_text}' (Sim: {max_similarity:.4f})")
    print(f"Reconstructed Text (Closest Match): '{best_match_text}' (Sim: {max_similarity:.4f})")
    METRICS_TRACKER.log_metric("Text Reconstruction Similarity", max_similarity)

    # --- MODULE 2: Personalized AI Thought Decryptor ("Brainprint") ---
    brainprint_nn = None
    training_losses_nn = []
    target_q_emb_for_stress, agg_ct_feat_for_stress, correct_bp_for_stress = None, None, None

    if nn:
        print("\n--- DEMO: Personalized AI Thought Decryptor (Brainprint NN) ---")
        EXPLAINER.add_explanation("Demo Section Start", "Brainprint NN")
        
        nn_sample_plain_embeddings = [] 
        for t in SAMPLE_TEXTS:
            emb_s = text_embedder_global.transform([t]); emb_d_raw = emb_s.toarray().flatten()
            if len(emb_d_raw) < NUM_QUBITS: emb_d = np.pad(emb_d_raw, (0, NUM_QUBITS - len(emb_d_raw)), 'constant')
            elif len(emb_d_raw) > NUM_QUBITS: emb_d = emb_d_raw[:NUM_QUBITS]
            else: emb_d = emb_d_raw
            nn_sample_plain_embeddings.append(emb_d)
        nn_sample_brainprints = [np.random.rand(BRAINPRINT_DIM).astype(np.float32) for _ in range(len(SAMPLE_TEXTS))]

        nn_input_ct_dim = LATTICE_SIM_SECURITY_PARAM + 1 
        brainprint_nn = BrainprintDecryptorNN(input_dim=nn_input_ct_dim, brainprint_dim=BRAINPRINT_DIM, output_dim=NUM_QUBITS)
        EXPLAINER.add_explanation("Brainprint NN Init", f"NN input (AggCTFeat {nn_input_ct_dim}D + Brainprint {BRAINPRINT_DIM}D) -> Output {NUM_QUBITS}D.")

        if q_encoder_decoder.pqc_circuit:
            METRICS_TRACKER.start_timer("M2_Brainprint_NN_Training")
            training_losses_nn = train_brainprint_decryptor(brainprint_nn, lattice_crypto, q_encoder_decoder, 
                                   nn_sample_plain_embeddings, nn_sample_brainprints, 
                                   NN_EPOCHS, NN_LEARNING_RATE, EXPLAINER, METRICS_TRACKER)
            METRICS_TRACKER.stop_timer("M2_Brainprint_NN_Training")
            if plt and training_losses_nn:
                METRICS_TRACKER.plot_line_data(training_losses_nn, title="Brainprint NN Training Loss", 
                                               xlabel="Epoch", ylabel="MSE Loss", filename_suffix="_nn_loss")

            test_idx = 0 
            test_plaintext_emb = nn_sample_plain_embeddings[test_idx]
            correct_bp_for_stress = nn_sample_brainprints[test_idx]
            test_wrong_bp = np.random.rand(BRAINPRINT_DIM).astype(np.float32)
            while np.allclose(test_wrong_bp, correct_bp_for_stress): test_wrong_bp = np.random.rand(BRAINPRINT_DIM).astype(np.float32)

            target_q_emb_for_stress = q_encoder_decoder.encode(test_plaintext_emb)
            A_shared_test = pk['A'] # Use A from previous general keygen
            sk_s_test = _derive_lattice_sk_from_brainprint(correct_bp_for_stress, lattice_crypto.n, lattice_crypto.q)
            e_pk_test = np.random.normal(0, 2, size=(lattice_crypto.m, 1)).astype(int)
            P_pk_test = (A_shared_test @ sk_s_test + e_pk_test) % lattice_crypto.q
            pk_nn_test = {'A': A_shared_test, 'P': P_pk_test}
            
            test_ciphertexts_nn = lattice_crypto.encrypt(pk_nn_test, target_q_emb_for_stress)
            agg_ct_feat_for_stress = aggregate_ciphertext_features(test_ciphertexts_nn, nn_input_ct_dim)
            
            brainprint_nn.eval() 
            with torch.no_grad():
                # Normalize aggregated ciphertext features for NN input
                pred_correct_tensor = brainprint_nn(torch.tensor(agg_ct_feat_for_stress / LATTICE_SIM_MODULUS, dtype=torch.float32), torch.tensor(correct_bp_for_stress, dtype=torch.float32))
                pred_q_emb_correct = pred_correct_tensor.cpu().numpy().flatten()
                pred_wrong_tensor = brainprint_nn(torch.tensor(agg_ct_feat_for_stress / LATTICE_SIM_MODULUS, dtype=torch.float32), torch.tensor(test_wrong_bp, dtype=torch.float32))
                pred_q_emb_wrong = pred_wrong_tensor.cpu().numpy().flatten()

            sim_correct = cosine_similarity(target_q_emb_for_stress.reshape(1,-1), pred_q_emb_correct.reshape(1,-1))[0,0]
            sim_wrong = cosine_similarity(target_q_emb_for_stress.reshape(1,-1), pred_q_emb_wrong.reshape(1,-1))[0,0]
            print(f"NN Decryption with CORRECT BP. Similarity: {sim_correct:.4f}")
            print(f"NN Decryption with WRONG BP. Similarity: {sim_wrong:.4f}")
            METRICS_TRACKER.log_metric("NN Decrypt Sim (Correct BP)", sim_correct)
            METRICS_TRACKER.log_metric("NN Decrypt Sim (Wrong BP)", sim_wrong)
            if plt: METRICS_TRACKER.plot_bar_chart([sim_correct, sim_wrong], ["Correct BP", "Wrong BP"], "Brainprint NN Decryption Similarity")
        else: EXPLAINER.add_explanation("Brainprint NN", "Skipped: PQC not available.")
    else: print("\n--- SKIPPING: Brainprint NN (PyTorch not available) ---")

    # --- MODULE 3: Zero-Knowledge-Like Proof ---
    print("\n--- DEMO: Zero-Knowledge-Like Proof of Mind Matching ---")
    EXPLAINER.add_explanation("Demo Section Start", "ZKP of Mind Matching")
    METRICS_TRACKER.start_timer("M3_ZKP_Proof_Full")
    zkp_handler = ZKPLikeProof(EXPLAINER)
    zkp_secret_bp = nn_sample_brainprints[0] if 'nn_sample_brainprints' in locals() and nn_sample_brainprints else np.random.rand(BRAINPRINT_DIM).astype(np.float32)
    
    s, v, r, t = zkp_handler.prove_knowledge(zkp_secret_bp)
    c = zkp_handler.verifier_generates_challenge()
    z = zkp_handler.prover_generates_response(s, r, c)
    zkp_passed = zkp_handler.verify_proof(v, t, c, z)
    
    METRICS_TRACKER.stop_timer("M3_ZKP_Proof_Full")
    print(f"ZKP: Public ID (v part): ...{str(v)[-6:]}. Verification: {'PASSED' if zkp_passed else 'FAILED'}")
    METRICS_TRACKER.log_metric("ZKP Verification Result", 1.0 if zkp_passed else 0.0)

    # --- MODULE 4: Quantum Noise-Resilience Simulation ---
    if qml and q_encoder_decoder.pqc_circuit:
        print("\n--- DEMO: Quantum Noise-Resilience Simulation ---")
        EXPLAINER.add_explanation("Demo Section Start", "Quantum Noise Resilience")
        METRICS_TRACKER.start_timer("M4_Noise_Simulation")
        noise_sim = QuantumNoiseSimulator(NUM_QUBITS, EXPLAINER, METRICS_TRACKER)
        ideal_qf, noisy_qf, fidelity = noise_sim.simulate_noise_impact(q_encoder_decoder, text_embedding_original, 
            NOISE_LEVEL_DEPOLARIZING, NOISE_LEVEL_AMPLITUDE_DAMPING, ERROR_MITIGATION_SHOTS)
        METRICS_TRACKER.stop_timer("M4_Noise_Simulation")

        if ideal_qf is not None and noisy_qf is not None and fidelity is not None:
            print(f"Noise Sim: Ideal QF: {np.round(ideal_qf[:3],3)} | Noisy QF: {np.round(noisy_qf[:3],3)}")
            print(f"Noise Sim: State Fidelity (Ideal vs Noisy): {fidelity:.4f}")
            output_sim_ideal_noisy = cosine_similarity(ideal_qf.reshape(1,-1), noisy_qf.reshape(1,-1))[0,0]
            METRICS_TRACKER.log_metric("Output QF Sim (Ideal vs Noisy)", output_sim_ideal_noisy)
            print(f"Noise Sim: Cosine Sim (Ideal QF vs Noisy QF): {output_sim_ideal_noisy:.4f}")
            if plt: METRICS_TRACKER.plot_bar_chart([1.0, fidelity, output_sim_ideal_noisy], 
                ["Ideal State (Self-Fid)", "State Fid (Ideal-Noisy)", "Output QF Sim (Ideal-Noisy)"], "Quantum Noise Impact")
    else: print("\n--- SKIPPING: Quantum Noise-Resilience (Pennylane or PQC inactive) ---")

    # --- MODULE 5: Adaptive PQC Optimization ---
    if qml and q_encoder_decoder.pqc_circuit:
        print("\n--- DEMO: Adaptive Quantum Circuit Architecture Optimization ---")
        EXPLAINER.add_explanation("Demo Section Start", "Adaptive PQC Optimization")
        METRICS_TRACKER.start_timer("M5_PQC_Optimization")
        pqc_optimizer = PQCAutomatedOptimizer(EXPLAINER, METRICS_TRACKER)
        print(f"PQC Opt: Initial depth {q_encoder_decoder.num_layers}.")
        optimized_depth = pqc_optimizer.optimize_pqc_depth(text_embedding_original, NUM_QUBITS, potential_depths=[1, 2, 3, 4])
        METRICS_TRACKER.stop_timer("M5_PQC_Optimization")
        print(f"PQC Opt: Optimized depth found (heuristic): {optimized_depth}.")
    else: print("\n--- SKIPPING: Adaptive PQC Optimization (Pennylane or PQC inactive) ---")

    # --- MODULE 6: Multi-Modal Thought Fusion ---
    if qml and q_encoder_decoder.pqc_circuit:
        print("\n--- DEMO: Multi-Modal Thought Fusion ---")
        EXPLAINER.add_explanation("Demo Section Start", "Multi-Modal Thought Fusion")
        METRICS_TRACKER.start_timer("M6_MultiModal_Processing")
        if TEXT_FEATURE_DIM <= 0 or IMAGE_FEATURE_DIM <=0 or (TEXT_FEATURE_DIM + IMAGE_FEATURE_DIM != NUM_QUBITS):
            EXPLAINER.add_explanation("MultiModal Fusion", "Skipped: Feature dimensions misconfigured.")
        else:
            text_embedder_mm = get_text_embedder(SAMPLE_TEXTS, TEXT_FEATURE_DIM)
            mm_encoder = MultiModalEncoder(text_embedder_mm, TEXT_FEATURE_DIM, IMAGE_FEATURE_DIM, EXPLAINER)
            fused_emb = mm_encoder.get_fused_embedding(SAMPLE_TEXTS[1], DUMMY_IMAGE_RAW)
            print(f"MultiModal: Fused embedding ({len(fused_emb)}D): {np.round(fused_emb[:3],3)}...")
            if len(fused_emb) == q_encoder_decoder.num_qubits:
                qf_mm = q_encoder_decoder.encode(fused_emb)
                enc_qf_mm = lattice_crypto.encrypt(pk, qf_mm)
                dec_qf_mm = lattice_crypto.decrypt(sk, enc_qf_mm)
                mm_sim = cosine_similarity(qf_mm.reshape(1,-1), dec_qf_mm.reshape(1,-1))[0,0]
                print(f"MultiModal: Decrypted Fused Features (Sim to original QF: {mm_sim:.4f})")
                METRICS_TRACKER.log_metric("MultiModal Decryption Similarity", mm_sim)
        METRICS_TRACKER.stop_timer("M6_MultiModal_Processing")
    else: print("\n--- SKIPPING: Multi-Modal Fusion (Pennylane or PQC inactive) ---")

    # --- MODULE 8: Blockchain Ledger ---
    print("\n--- DEMO: Blockchain-Based Thought Ledger ---")
    EXPLAINER.add_explanation("Demo Section Start", "Blockchain Thought Ledger")
    METRICS_TRACKER.start_timer("M8_Blockchain_Operations")
    thought_ledger = Blockchain(explainer=EXPLAINER)
    if encrypted_q_features:
        try:
            thought_log_str = json.dumps(encrypted_q_features, sort_keys=True)
            hashed_thought = hashlib.sha256(thought_log_str.encode()).hexdigest()
            thought_ledger.add_block(hashed_thought)
            if 'enc_qf_mm' in locals() and enc_qf_mm: 
                 thought_log_str_2 = json.dumps(enc_qf_mm, sort_keys=True)
                 hashed_thought_2 = hashlib.sha256(thought_log_str_2.encode()).hexdigest()
                 thought_ledger.add_block(hashed_thought_2)
            else: thought_ledger.add_block(hashlib.sha256("dummy_thought_2".encode()).hexdigest())
            thought_ledger.print_chain()
            METRICS_TRACKER.log_metric("Blockchain Valid", 1.0 if thought_ledger.is_chain_valid() else 0.0)
        except TypeError as e: EXPLAINER.add_explanation("Blockchain Error", f"Serialization: {e}")
    METRICS_TRACKER.stop_timer("M8_Blockchain_Operations")

    # --- MODULE 10: Security Stress Test ---
    if nn and brainprint_nn and target_q_emb_for_stress is not None and agg_ct_feat_for_stress is not None:
        print("\n--- DEMO: Security Stress Test ---")
        EXPLAINER.add_explanation("Demo Section Start", "Security Stress Test")
        METRICS_TRACKER.start_timer("M10_Security_Stress_Test")
        correct_sim_stress, wrong_sims_stress = run_security_stress_test(brainprint_nn, target_q_emb_for_stress, 
                                 agg_ct_feat_for_stress, correct_bp_for_stress,
                                 num_wrong_attempts=5, explainer=EXPLAINER, metrics_tracker=METRICS_TRACKER)
        METRICS_TRACKER.stop_timer("M10_Security_Stress_Test")
        if plt and wrong_sims_stress is not None and correct_sim_stress is not None:
            plot_labels = [f"WrongBP{i+1}" for i in range(len(wrong_sims_stress))] + ["CorrectBP"]
            plot_values = wrong_sims_stress + [correct_sim_stress]
            METRICS_TRACKER.plot_bar_chart(plot_values, plot_labels, "Security Stress Test Similarities", ylabel="Cosine Similarity")
    else: print("\n--- SKIPPING: Security Stress Test (NN components/data unavailable) ---")

    # --- Final Reports ---
    print("\n--- FINAL REPORTS ---")
    summary_report_text = EXPLAINER.get_summary_report()
    print(summary_report_text)
    try:
        with open("explainability_report.txt", "w", encoding="utf-8") as f: f.write(summary_report_text)
        print("Explainability report saved to explainability_report.txt")
    except Exception as e: print(f"Error saving explainability report: {e}")
    
    METRICS_TRACKER.display_summary()
    generate_research_proposal_md()

    print("\n" + "="*50 + "\nQuantum-Neuro AI Sovereign 2.0 Demo Finished\n" + "="*50)

if __name__ == "__main__":
    if not qml: print("CRITICAL WARNING: PennyLane not installed. Quantum features will be limited.")
    if not nn: print("WARNING: PyTorch not installed. Neural network features will be skipped.")
    main_demo()