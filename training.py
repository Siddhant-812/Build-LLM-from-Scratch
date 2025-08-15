import torch
import torch.nn as nn
from torch.optim import AdamW
from data_utils import create_dataloader

def train_model(model, raw_text, tokenizer, num_epochs, learning_rate, batch_size, context_length, device,save_path="model_weights.pth"):
    """
    The main training loop for the GPT model.
    Now creates the dataloader inside the loop for each epoch.
    """
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # --- THE FIX: Recreate the dataloader for each epoch ---
        dataloader = create_dataloader(raw_text, batch_size, context_length, tokenizer)
        
        total_loss = 0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # This check prevents division by zero if the dataloader was somehow empty
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | No batches were processed.")

    print("\nTraining complete!")
    print(f"Saving model weights to {save_path}")
    torch.save(model.state_dict(), save_path)