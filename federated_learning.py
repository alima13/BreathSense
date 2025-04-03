#=============================================
# BreathSense: Just a project to test the idea of having privacy gurantees within Raspiratory Sounds Schema. 
    # "Federated Learning for Respiratory Sound Analysis",
    # "Discovering Attention Methods with ICBHI Respiratory Sound Database",
    # Author: Ali Mahdavi,
    # Disclaimer: The codes will undergo massive changes during the time,
    # Dataset: https://bhichallenge.med.auth.gr/node/51,
    # Current Steps: 
        ## Create spectrograms from respiratory audio files
        ## Build and train a federated learning model with attention mechanisms (CNN-LSTM model architecture for respiratory sound classification)
        ## Adding LSH/ Ensuring Privacy Gurantees
    # If you want to give the code to AI for any enhancements, please note that in the hashing part it maight get nasty. 
#=============================================

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import datasketch
from typing import List, Dict, Tuple, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RespiratoryDataset(Dataset):
    def __init__(self, feature_list, labels):
        self.features = feature_list
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

class RespiratoryAcousticModel(nn.Module):
    
    def __init__(self, input_dim=24, hidden_dim=64, num_classes=4):
        super(RespiratoryAcousticModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        
        # LSTM for temporal dynamics
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        
        return out

class LSHAggregator:
    
    def __init__(self, num_perm=128, threshold=0.7):
        self.num_perm = num_perm
        self.threshold = threshold

    def _model_to_vector(self, model: nn.Module) -> np.ndarray:
        param_vector = []
        for param in model.parameters():
            param_vector.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(param_vector)

    def _vector_to_model(self, vector: np.ndarray, model_template: nn.Module) -> nn.Module:
        model = copy.deepcopy(model_template)
        pos = 0
        for param in model.parameters():
            num_param = param.numel()
            param.data = torch.from_numpy(vector[pos:pos+num_param].reshape(param.data.shape)).to(param.device)
            pos += num_param
        return model

    def compute_hash(self, model: nn.Module) -> datasketch.MinHash:
        """Compute MinHash of model parameters."""
        model_vector = self._model_to_vector(model)
        model_vector = np.round(model_vector * 1000)
        minhash = datasketch.MinHash(num_perm=self.num_perm)
        for i, val in enumerate(model_vector):
            item = f"{i}:{val}"
            minhash.update(item.encode('utf-8'))
        return minhash

    def cluster_models(self, models: list) -> dict:
        """Cluster models based on LSH similarity."""
        model_hashes = [self.compute_hash(model) for model in models]
        lsh_index = datasketch.MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for i, minhash in enumerate(model_hashes):
            lsh_index.insert(i, minhash)
        clusters = {}
        processed = set()
        for i, minhash in enumerate(model_hashes):
            if i in processed:
                continue
            similar = lsh_index.query(minhash)
            if similar:
                cluster_id = len(clusters)
                clusters[cluster_id] = similar
                processed.update(similar)
        return clusters

    def aggregate_models(self, models: list, weights: list = None) -> nn.Module:
        """Aggregate models using LSH-based clustering."""
        if len(models) == 0:
            raise ValueError("No models to aggregate")
        if len(models) == 1:
            return copy.deepcopy(models[0])
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        clusters = self.cluster_models(models)
        cluster_models = []
        cluster_weights = []
        for cluster_id, model_indices in clusters.items():
            cluster_model_list = [models[i] for i in model_indices]
            cluster_weight_list = [weights[i] for i in model_indices]
            total_weight = sum(cluster_weight_list)
            if total_weight > 0:
                cluster_weight_list = [w / total_weight for w in cluster_weight_list]
            else:
                cluster_weight_list = [1.0 / len(cluster_model_list)] * len(cluster_model_list)
            
            avg_model = self._average_models(cluster_model_list, cluster_weight_list)
            cluster_models.append(avg_model)
            cluster_weights.append(sum([weights[i] for i in model_indices]))
        
        total_cluster_weight = sum(cluster_weights)
        if total_cluster_weight > 0:
            cluster_weights = [w / total_cluster_weight for w in cluster_weights]
        else:
            cluster_weights = [1.0 / len(cluster_models)] * len(cluster_models)
        
        final_model = self._average_models(cluster_models, cluster_weights)
        return final_model

    def _average_models(self, models: list, weights: list) -> nn.Module:
        """Average model parameters with weights."""
        model_vectors = [self._model_to_vector(model) for model in models]
        avg_vector = np.zeros_like(model_vectors[0])
        for vec, weight in zip(model_vectors, weights):
            avg_vector += vec * weight
        return self._vector_to_model(avg_vector, models[0])

class FederatedLearning:
    """Federated Learning with LSH-based aggregation."""
    
    def __init__(self, model_fn, num_clients=10, client_fraction=0.5,
                 local_epochs=5, learning_rate=0.001, lsh_threshold=0.7):
        self.clients = []
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.global_model = model_fn().to(DEVICE)
        self.aggregator = LSHAggregator(threshold=lsh_threshold)
        self.train_losses = []
        self.val_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def add_client(self, features, labels):
        dataset = RespiratoryDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        local_model = copy.deepcopy(self.global_model)
        self.clients.append({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'model': local_model
        })

    def select_clients(self):
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        selected_clients = np.random.choice(range(len(self.clients)), size=num_selected, replace=False)
        return selected_clients

    def train_client(self, client_idx):
        """Train a client's local model."""
        client = self.clients[client_idx]
        model = client['model']
        train_loader = client['train_loader']
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        epoch_loss = 0
        for epoch in range(self.local_epochs):
            for features, labels in train_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
        client['model'] = model
        return model, epoch_loss

    def evaluate(self, model, val_loader):
        """Evaluate model on validation data."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return {
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_round(self):
        """Run a round of federated training."""
        selected_clients = self.select_clients()
        trained_models = []
        client_losses = []
        for client_idx in selected_clients:
            self.clients[client_idx]['model'].load_state_dict(self.global_model.state_dict())
            trained_model, loss = self.train_client(client_idx)
            trained_models.append(trained_model)
            client_losses.append(loss)
        weights = [1.0 / len(trained_models)] * len(trained_models)
        self.global_model = self.aggregator.aggregate_models(trained_models, weights)
        avg_train_loss = sum(client_losses) / len(client_losses)
        self.train_losses.append(avg_train_loss)
        return avg_train_loss

    def evaluate_global_model(self):
        """Evaluate global model on all clients' validation data."""
        metrics_sum = {
            'loss': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        for client in self.clients:
            client_metrics = self.evaluate(self.global_model, client['val_loader'])
            for key in metrics_sum:
                metrics_sum[key] += client_metrics[key]
        avg_metrics = {k: v / len(self.clients) for k, v in metrics_sum.items()}
        for key in self.val_metrics:
            self.val_metrics[key].append(avg_metrics[key])
        return avg_metrics

    def train(self, num_rounds=10):
        for round_idx in range(num_rounds):
            train_loss = self.train_round()
            val_metrics = self.evaluate_global_model()
            print(f"Round {round_idx+1}/{num_rounds}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        return self.global_model

# Example usage 
if __name__ == "__main__":
    # Load features and labels
    extractor = FeatureExtractor()
    features = []
    labels = []
    for audio_file, label in zip(["file1.wav", "file2.wav"], [0, 1], [0, 1]):
        feature = extractor.extract_features(audio_file)
        if feature is not None:
            features.append(feature)
            labels.append(label)

    # Initialize federated learning setup -- Playing with these paramethers to optimize the whole set
    num_clients = 2
    federated_learner = FederatedLearning(
        model_fn=RespiratoryAcousticModel,
        num_clients=num_clients,
        client_fraction=1.0,
        local_epochs=2,
        learning_rate=0.002,
        lsh_threshold=0.8
    )

    # Add clients with their local data
    for i in range(num_clients):
        federated_learner.add_client(features, labels)

    # Train the federated model
    global_model = federated_learner.train(num_rounds=5)

    # Evaluate the trained global model
    val_metrics = federated_learner.evaluate_global_model()
    print(f"Final Validation Metrics: Loss: {val_metrics['loss']:.4f}, "
          f"Accuracy: {val_metrics['accuracy']:.4f}, "
          f"F1 Score: {val_metrics['f1']:.4f}")

