import torch
import torch.nn as nn
from torch.optim import AdamW
import requests
import zipfile
import io
import os

from model import GPTModel, GPTClassificationModel
from tokenizer import BPETokenizer

# --- Configuration ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
CONTEXT_LENGTH = 64
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "model_weights.pth"
FINETUNED_WEIGHTS_PATH = "finetuned_classifier_weights.pth" # New path for the finetuned model

# Finetuning Config
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-5
NUM_CLASSES = 2

def download_and_prepare_data():
    # ... (this function remains exactly the same) ...
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
    print("Downloading sentiment dataset...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open('sentiment labelled sentences/amazon_cells_labelled.txt') as f:
        lines = f.read().decode('utf-8').strip().split('\n')
    texts = [line.split('\t')[0] for line in lines]
    labels = [int(line.split('\t')[1]) for line in lines]
    return texts, labels

if __name__ == '__main__':
    # ... (Steps 1, 2, 3, and 4 for loading models and data are the same) ...
    # 1. Load tokenizer
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(file_url).text
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(raw_text, VOCAB_SIZE)

    # 2. Load pretrained base model
    base_model = GPTModel(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, CONTEXT_LENGTH)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Pretrained weights not found at {WEIGHTS_PATH}. Please run main.py first.")
    base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("Successfully loaded pretrained model weights.")

    # 3. Create classification model
    classification_model = GPTClassificationModel(base_model, NUM_CLASSES).to(DEVICE)

    # 4. Prepare dataset
    texts, labels = download_and_prepare_data()
    encoded_texts = [bpe_tokenizer.encode(text) for text in texts]
    max_len = CONTEXT_LENGTH
    padded_texts = torch.zeros(len(encoded_texts), max_len, dtype=torch.long)
    for i, tokens in enumerate(encoded_texts):
        seq_len = min(len(tokens), max_len)
        padded_texts[i, :seq_len] = torch.tensor(tokens[:seq_len])
    labels = torch.tensor(labels)

    # 5. The Finetuning Loop
    optimizer = AdamW(classification_model.parameters(), lr=FINETUNE_LR)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nStarting classification finetuning...")
    classification_model.train()
    for epoch in range(FINETUNE_EPOCHS):
        inputs, targets = padded_texts.to(DEVICE), labels.to(DEVICE)
        logits = classification_model(inputs)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | Loss: {loss.item():.4f}")

    print("\nFinetuning complete!")

    # --- THE FIX: Save the finetuned model's weights ---
    print(f"Saving finetuned model weights to {FINETUNED_WEIGHTS_PATH}")
    torch.save(classification_model.state_dict(), FINETUNED_WEIGHTS_PATH)

    # 6. Test the finetuned model
    classification_model.eval()
    with torch.no_grad():
        test_sentence = "The battery life is amazing."
        encoded = bpe_tokenizer.encode(test_sentence)
        padded = torch.zeros(1, max_len, dtype=torch.long)
        padded[0, :len(encoded)] = torch.tensor(encoded)
        
        logits = classification_model(padded.to(DEVICE))
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"\nTest Sentence: '{test_sentence}'")
        print(f"Predicted Sentiment: {sentiment}")