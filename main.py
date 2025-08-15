import torch
import os # Import the os module to check for files
from data_loader import download_and_read_text
from tokenizer import BPETokenizer
from model import GPTModel
from training import train_model
from generation import generate_text

# --- Configuration ---
# (All configuration variables remain the same)
VOCAB_SIZE = 512
BATCH_SIZE = 4
CONTEXT_LENGTH = 64
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "model_weights.pth"

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load data and train tokenizer (we always need the tokenizer)
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = download_and_read_text(file_url)
    bpe_tokenizer = BPETokenizer()
    if raw_text:
        bpe_tokenizer.train(raw_text, VOCAB_SIZE)

    # 2. Instantiate the GPT Model
    gpt_model = GPTModel(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH
    )
    gpt_model.to(DEVICE) # Move model to device early

    # 3. --- NEW: Check for saved weights ---
    if os.path.exists(WEIGHTS_PATH):
        print(f"Found saved weights. Loading from {WEIGHTS_PATH}")
        gpt_model.load_state_dict(torch.load(WEIGHTS_PATH))
    else:
        print("No saved weights found. Starting training...")
        train_model(
            model=gpt_model,
            raw_text=raw_text,
            tokenizer=bpe_tokenizer,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            device=DEVICE,
            save_path=WEIGHTS_PATH
        )

    # 4. Generate new text
    print("\n" + "="*80)
    print("Generating new text with Temperature=0.8 and Top-K=50...")
    print("="*80)
    
    start_context = "O Romeo, Romeo, wherefore art thou Romeo?"
    
    generated_text = generate_text(
        model=gpt_model,
        tokenizer=bpe_tokenizer,
        context=start_context,
        max_new_tokens=100,
        context_length=CONTEXT_LENGTH,
        device=DEVICE,
        temperature=0.8, # Make the model a bit more conservative
        top_k=50         # Only consider the 50 most likely next words
    )
    
    print(generated_text)
    print("\n" + "="*80)