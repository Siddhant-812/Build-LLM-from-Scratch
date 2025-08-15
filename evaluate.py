import torch
import torch.nn as nn
import os
import requests
import math

from model import GPTModel
from tokenizer import BPETokenizer

# --- Configuration ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
CONTEXT_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "model_weights.pth"

def calculate_perplexity(model, text, tokenizer, device):
    """
    Calculates the perplexity of a model on a given text.
    """
    model.eval()
    
    # 1. Encode the entire text
    tokens = tokenizer.encode(text)
    token_ids = torch.tensor([tokens], device=device)
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # 2. Iterate through the text in chunks of context_length
        for i in range(0, token_ids.size(1) - CONTEXT_LENGTH - 1, CONTEXT_LENGTH):
            inputs = token_ids[:, i:i+CONTEXT_LENGTH]
            targets = token_ids[:, i+1:i+CONTEXT_LENGTH+1]

            logits = model(inputs)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
    if num_batches == 0:
        return float('inf') # Cannot calculate if text is too short

    # 3. Calculate perplexity
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return perplexity

if __name__ == '__main__':
    # 1. Load tokenizer (trained on Shakespeare)
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(file_url).text
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(raw_text, VOCAB_SIZE)

    # 2. Load the pretrained model
    model = GPTModel(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, CONTEXT_LENGTH).to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Pretrained weights not found at {WEIGHTS_PATH}.")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("Successfully loaded pretrained model weights.")

    # 3. Download a new, unseen test dataset
    print("\nDownloading test dataset (The Wonderful Wizard of Oz)...")
    test_url = "https://www.gutenberg.org/files/55/55-0.txt"
    test_text = requests.get(test_url).text
    
    # 4. Calculate and print perplexity
    ppl = calculate_perplexity(model, test_text, bpe_tokenizer, DEVICE)
    
    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    print(f"Perplexity on 'The Wonderful Wizard of Oz': {ppl:.2f}")
    print("\nExplanation:")
    print("A lower perplexity is better. This score gives us a quantitative measure")
    print("of how well our model, trained on Shakespeare, understands a new text.")
    print("A high score indicates the model was very 'surprised' by the new style and vocabulary.")
    print("="*80)