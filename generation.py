import torch

def generate_text(model, tokenizer, context, max_new_tokens, context_length, device, temperature=1.0, top_k=None):
    """
    Generates new text with temperature scaling and top-k sampling.
    """
    model.eval()
    
    token_ids = tokenizer.encode(context)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_length:]

        with torch.no_grad():
            logits = model(x_cond)
        
        logits = logits[:, -1, :]

        # --- NEW: Apply Temperature Scaling ---
        if temperature > 0:
            logits = logits / temperature

        # --- NEW: Apply Top-K Sampling ---
        if top_k is not None:
            # Get the top k logits and their indices
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            # Create a new tensor full of -inf
            filtered_logits = torch.full_like(logits, float('-inf'))
            # Place the top k logits back into the new tensor
            filtered_logits.scatter_(1, top_k_indices, top_k_logits)
            logits = filtered_logits
        
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token_id), dim=1)

    generated_ids = x.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)