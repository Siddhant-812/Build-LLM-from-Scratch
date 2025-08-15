import torch
import torch.nn as nn
from torch.optim import AdamW
import os
import requests
from datasets import load_dataset

import bitsandbytes.nn as bnb

from model import GPTModel
from tokenizer import BPETokenizer
from lora import LoRALayer
from generation import generate_text

# --- Configuration ---
VOCAB_SIZE = 512
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 3
CONTEXT_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "model_weights.pth"
QLORA_WEIGHTS_PATH = "qlora_instruction_weights.pth"

# QLoRA Config
LORA_RANK = 8
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-4

# --- Helper Function ---
def format_instruction(sample):
    return (f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Context:\n{sample['context']}\n\n"
            f"### Response:\n{sample['response']}")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemError("QLoRA requires a CUDA-enabled GPU.")

    # 1. Load both datasets first
    print("Loading datasets...")
    shakespeare_text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
    instruction_dataset = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(500))
    formatted_texts = [format_instruction(sample) for sample in instruction_dataset]

    # 2. Train tokenizer on the combined text
    print("Training tokenizer on combined Shakespeare and Dolly dataset...")
    combined_text = shakespeare_text + "\n" + "\n".join(formatted_texts)
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(combined_text, VOCAB_SIZE)
    
    # 3. --- THE DEFINITIVE FIX: Data Sanitization ---
    print("Encoding and sanitizing the instruction dataset...")
    encoded_texts_sane = []
    for text in formatted_texts:
        tokens = bpe_tokenizer.encode(text)
        # Check if any token ID is out of the vocabulary range
        if all(t < VOCAB_SIZE for t in tokens):
            encoded_texts_sane.append(tokens)
        else:
            print(f"Warning: Discarding a sample because it contains out-of-vocabulary tokens.")
    
    print(f"Original dataset size: {len(formatted_texts)}")
    print(f"Sanitized dataset size: {len(encoded_texts_sane)}")


    # 4. Load the base GPT model
    base_model = GPTModel(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE, CONTEXT_LENGTH)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Pretrained weights not found at {WEIGHTS_PATH}.")
    base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    print("Successfully loaded pretrained model weights.")

    # 5. Apply 4-bit quantization
    for block in base_model.blocks:
        for head in block.attn.heads:
            head.q_proj = bnb.Linear4bit(head.q_proj.in_features, head.q_proj.out_features, bias=False)
            head.k_proj = bnb.Linear4bit(head.k_proj.in_features, head.k_proj.out_features, bias=False)
            head.v_proj = bnb.Linear4bit(head.v_proj.in_features, head.v_proj.out_features, bias=False)
        block.ff.net[0] = bnb.Linear4bit(block.ff.net[0].in_features, block.ff.net[0].out_features, bias=True)
        block.ff.net[2] = bnb.Linear4bit(block.ff.net[2].in_features, block.ff.net[2].out_features, bias=True)
    print("\nApplied 4-bit quantization to the model.")

    # 6. Freeze parameters and inject LoRA layers
    for param in base_model.parameters():
        param.requires_grad = False
    for block in base_model.blocks:
        block.attn.out_proj = LoRALayer(block.attn.out_proj, rank=LORA_RANK)
    model = base_model.to(DEVICE)
    
    # ... (The rest of the script is the same, but it uses the sanitized `encoded_texts_sane`) ...
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (QLoRA): {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=FINETUNE_LR)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nStarting QLoRA instruction finetuning...")
    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0
        # Use the sanitized list for training
        for tokens in encoded_texts_sane:
            if len(tokens) < 2: continue
            # Truncate long sequences to fit context length
            if len(tokens) > CONTEXT_LENGTH + 1:
                tokens = tokens[:CONTEXT_LENGTH + 1]
            
            inputs = torch.tensor([tokens[:-1]], device=DEVICE)
            targets = torch.tensor([tokens[1:]], device=DEVICE)

            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(encoded_texts_sane)
        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | Average Loss: {avg_loss:.4f}")

    print("\nFinetuning complete!")

    # ... (Saving and Testing are the same) ...
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(trainable_state_dict, QLORA_WEIGHTS_PATH)
    print(f"Saved QLoRA adapter weights to {QLORA_WEIGHTS_PATH}")
    
    model.eval()
    prompt = format_instruction({"instruction": "What is the capital of Italy?", "context": "", "response": ""})
    print("\n--- Generating Response ---")
    generated_text = generate_text(model, bpe_tokenizer, prompt, 50, CONTEXT_LENGTH, DEVICE)
    print(generated_text)