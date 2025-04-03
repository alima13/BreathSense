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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from feature_extraction import FeatureExtractor

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
    """CNN-LSTM model for respiratory sound classification."""
    
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

# Example usage
if __name__ == "__main__":
    # Load features and labels
    extractor = FeatureExtractor()
    features = []
    labels = []
    # Assuming you have a list of audio files and corresponding labels
    for audio_file, label in zip(["file1.wav", "file2.wav"], [0, 1]):
        feature = extractor.extract_features(audio_file)
        if feature is not None:
            features.append(feature)
            labels.append(label)
    
    # Create dataset and dataloader
    dataset = RespiratoryDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model and optimizer
    model = RespiratoryAcousticModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    for epoch in range(10):
        model.train()
        epoch_loss = 0
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
        
        # Evaluate on validation set
        model.eval()
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
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}")
