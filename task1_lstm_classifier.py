import os
import json
import torch
import pickle
import pretty_midi
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = os.path.join("task1_composer_classification")
MIDI_DIR = os.path.join(DATA_DIR, "midis")
MAX_SEQ_LEN = 512
BATCH_SIZE = 32
EPOCHS = 20
EMBED_DIM = 32
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset Definition ---
class MidiComposerDataset(Dataset):
    def __init__(self, paths, labels=None, label_encoder=None):
        self.paths = paths
        self.labels = labels
        self.label_encoder = label_encoder

    def extract_pitch_sequence(self, file_path):
        try:
            midi = pretty_midi.PrettyMIDI(file_path)
            pitches = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    pitches += [note.pitch for note in instrument.notes]
            pitches = pitches[:MAX_SEQ_LEN]
            if not pitches:
                pitches = [0]
        except:
            pitches = [0]

        # Pad sequence
        padded = pitches + [0] * (MAX_SEQ_LEN - len(pitches))
        return torch.tensor(padded[:MAX_SEQ_LEN], dtype=torch.long)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sequence = self.extract_pitch_sequence(path)
        if self.labels:
            label = self.label_encoder.transform([self.labels[idx]])[0]
            return sequence, torch.tensor(label)
        return sequence, path


# --- LSTM Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[0], hn[1]), dim=1)
        return self.fc(out)


# --- Training Pipeline ---
def train_model():
    with open(os.path.join(DATA_DIR, "train.json"), "r") as f:
        train_dict = json.load(f)

    file_paths = [os.path.join(MIDI_DIR, k) for k in train_dict.keys()]
    labels = list(train_dict.values())

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_classes = len(label_encoder.classes_)

    train_dataset = MidiComposerDataset(file_paths, labels, label_encoder)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMClassifier(128, EMBED_DIM, HIDDEN_DIM, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "model1_lstm.pt")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("[✓] Training complete, model and encoder saved.")


# --- Prediction Pipeline ---
def predict():
    with open(os.path.join(DATA_DIR, "test.json"), "r") as f:
        test_files = json.load(f)

    test_paths = [os.path.join(DATA_DIR, f) for f in test_files]

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)

    model = LSTMClassifier(128, EMBED_DIM, HIDDEN_DIM, num_classes).to(DEVICE)
    model.load_state_dict(torch.load("model1_lstm.pt"))
    model.eval()

    test_dataset = MidiComposerDataset(test_paths)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = {}
    with torch.no_grad():
        for x, path in tqdm(test_loader, desc="Predicting"):
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().item()
            label = label_encoder.inverse_transform([pred])[0]
            predictions[os.path.basename(path[0])] = label

    with open("predictions1.json", "w") as f:
        json.dump(predictions, f)
    print("[✓] predictions1.json saved!")


# --- Run from PyCharm ---
if __name__ == "__main__":
    train_model()
    predict()

