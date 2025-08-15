import torch
import torch.nn as nn
from torch.optim import AdamW
import os
import requests # Make sure to import requests

from model import GPTModel, GPTClassificationModel
from tokenizer import BPETokenizer
from lora import LoRALayer # Import our new LoRALayer
from finetune_classifier import download_and_prepare_data # Reuse our data function

# --- Configuration ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
CONTEXT_LENGTH = 64
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "model_weights.pth"
LORA_WEIGHTS_PATH = "lora_classifier_weights.pth" # Save LoRA weights here

# LoRA Config
LORA_RANK = 8 # The rank for our LoRA matrices. A smaller rank means fewer parameters.

# Finetuning Config
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4 # We can often use a slightly higher LR for LoRA

if __name__ == '__main__':
    # 1. Load tokenizer and data (same as before)
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(file_url).text
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(raw_text, VOCAB_SIZE)
    texts, labels = download_and_prepare_data()

    # 2. Load the pretrained base GPT model
    base_model = GPTModel(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, CONTEXT_LENGTH)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Pretrained weights not found at {WEIGHTS_PATH}.")
    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
    base_model.load_state_dict(state_dict)
    base_model.to(DEVICE)
    print("Successfully loaded pretrained model weights.")

    # --- LoRA Injection ---
    # 3. Freeze all parameters in the base model
    for param in base_model.parameters():
        param.requires_grad = False
        
    # 4. Inject LoRA layers into the output projection of each attention block
    for block in base_model.blocks:
        block.attn.out_proj = LoRALayer(block.attn.out_proj, rank=LORA_RANK)
    
    # 5. Create the classification model (the head will be trainable by default)
    classification_model = GPTClassificationModel(base_model, num_classes=2).to(DEVICE)

    # 6. Count and print the number of trainable parameters
    total_params = sum(p.numel() for p in classification_model.parameters())
    trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters (LoRA + classifier head): {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # 7. Prepare data (same as before)
    max_len = CONTEXT_LENGTH
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, tokens in enumerate(bpe_tokenizer.encode(text) for text in texts):
        seq_len = min(len(tokens), max_len)
        padded_texts[i, :seq_len] = torch.tensor(tokens[:seq_len])
    labels = torch.tensor(labels)

    # 8. The Finetuning Loop (trains only LoRA layers and the head)
    optimizer = AdamW(classification_model.parameters(), lr=FINETUNE_LR)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nStarting LoRA classification finetuning...")
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

    # 9. Save ONLY the LoRA parameters and the classifier head
    trainable_state_dict = {k: v for k, v in classification_model.state_dict().items() if v.requires_grad}
    torch.save(trainable_state_dict, LORA_WEIGHTS_PATH)
    print(f"Saved trainable (LoRA) weights to {LORA_WEIGHTS_PATH}")
    
    classification_model.eval()
    with torch.no_grad():
        test_sentence = "The battery life is good."
        encoded = bpe_tokenizer.encode(test_sentence)
        padded = torch.zeros(1, max_len, dtype=torch.long)
        padded[0, :len(encoded)] = torch.tensor(encoded)
        
        logits = classification_model(padded.to(DEVICE))
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"\nTest Sentence: '{test_sentence}'")
        print(f"Predicted Sentiment: {sentiment}")