import torch

def create_dataloader(text, batch_size, context_length, tokenizer):
    """
    Creates a DataLoader that yields batches of input (x) and target (y) tensors.
    """
    # 1. Encode the entire text dataset
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids)

    # 2. Calculate the number of batches we can create
    num_tokens = len(token_ids)
    num_batches = num_tokens // (batch_size * context_length)

    if num_batches == 0:
        raise ValueError("Dataset is too small to create a single batch. "
                         "Please use a larger text file or smaller batch/context size.")

    # 3. Trim the dataset to fit neatly into batches
    # This discards the last few tokens that don't make a full batch
    num_tokens_in_batches = num_batches * batch_size * context_length
    token_ids = token_ids[:num_tokens_in_batches]

    # 4. Reshape the tokens into batches
    # This creates a (batch_size, num_tokens_per_batch) tensor
    token_ids = token_ids.view(batch_size, -1)

    # 5. Yield batches of data
    for i in range(0, token_ids.size(1) - context_length, context_length):
        # The input is a chunk of the text
        x = token_ids[:, i:i+context_length]
        # The target is the next token in the sequence
        y = token_ids[:, i+1:i+context_length+1]
        yield x, y