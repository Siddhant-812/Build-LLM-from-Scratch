import torch
import torch.nn as nn

class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, context_length):
        """
        Initializes the embedding layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embeddings (and the model).
            context_length (int): The maximum length of an input sequence.
        """
        super().__init__()
        # Token embedding layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Positional embedding layer
        self.pos_emb = nn.Embedding(context_length, d_model)

    def forward(self, x):
        """
        Forward pass for the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined token and positional embeddings.
        """
        # Get the device from the input tensor
        device = x.device
        # Get the sequence length from the input tensor shape
        seq_len = x.shape[1]
        
        # Get positions for the positional embeddings
        positions = torch.arange(seq_len, device=device)
        
        # Get token and positional embeddings
        tok_embeddings = self.tok_emb(x)
        pos_embeddings = self.pos_emb(positions)
        
        # Add them together (broadcasting pos_embeddings across the batch)
        return tok_embeddings + pos_embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        """
        Initializes a single head of self-attention.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            d_head (int): The dimensionality of the query, key, and value vectors.
        """
        super().__init__()
        self.d_head = d_head
        # Linear layers to project input embeddings into Q, K, V
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x, mask=None):
        """
        Forward pass for a single self-attention head.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): A mask to prevent attention to future tokens. Defaults to None.

        Returns:
            torch.Tensor: The output of the attention head (batch_size, seq_len, d_head).
            torch.Tensor: The attention weights (batch_size, seq_len, seq_len).
        """
        # Project the input into query, key, and value vectors
        q = self.q_proj(x) # (batch_size, seq_len, d_head)
        k = self.k_proj(x) # (batch_size, seq_len, d_head)
        v = self.v_proj(x) # (batch_size, seq_len, d_head)

        # Calculate attention scores (dot product between queries and keys)
        # Transpose k to get (batch_size, d_head, seq_len) for matrix multiplication
        attn_scores = q @ k.transpose(-2, -1) # (batch_size, seq_len, seq_len)

        # Scale the scores
        attn_scores = attn_scores / (self.d_head ** 0.5)

        # Apply the mask if provided
        if mask is not None:
            # Set the masked values to a very small number so they become zero after softmax
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1) # (batch_size, seq_len, seq_len)

        # Multiply the weights by the values to get the output
        output = attn_weights @ v # (batch_size, seq_len, d_head)

        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Initializes the Multi-Head Attention block.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        # d_head is the dimension of each head's output
        self.d_head = d_model // n_heads

        # A list of all the individual attention heads
        self.heads = nn.ModuleList([
            SelfAttention(d_model, self.d_head) for _ in range(n_heads)
        ])
        # A final linear layer to project the concatenated outputs
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, x, mask=None):
        """
        Forward pass for the Multi-Head Attention block.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): The causal mask. Defaults to None.

        Returns:
            torch.Tensor: The final output of the multi-head attention block.
        """
        # 1. Run each attention head in parallel
        head_outputs = [head(x, mask)[0] for head in self.heads]

        # 2. Concatenate the outputs of all heads along the last dimension
        x = torch.cat(head_outputs, dim=-1) # (batch_size, seq_len, d_model)

        # 3. Pass the concatenated output through the final projection layer
        x = self.out_proj(x)

        return x
    
class FeedForward(nn.Module):
    """
    A simple feed-forward neural network block.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    The main Transformer Block.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        # The multi-head attention sub-layer
        self.attn = MultiHeadAttention(d_model, n_heads)
        # The feed-forward sub-layer
        self.ff = FeedForward(d_model, d_model * 4) # d_ff is typically 4 * d_model
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Block.
        """
        # First sub-layer: Attention with residual connection
        attn_output = self.attn(self.ln1(x), mask)
        x = x + attn_output

        # Second sub-layer: Feed-forward with residual connection
        ff_output = self.ff(self.ln2(x))
        x = x + ff_output

        return x
    
class GPTModel(nn.Module):
    """
    The full GPT model architecture.
    """
    def __init__(self, d_model, n_heads, n_layers, vocab_size, context_length):
        super().__init__()
        self.embedding = TokenPositionalEmbedding(vocab_size, d_model, context_length)
        # We use nn.ModuleList which is designed to hold a list of modules
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """
        Forward pass for the full GPT model.
        """
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        x = self.embedding(x)
        # Loop through each block and pass the output to the next
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    

class GPTClassificationModel(nn.Module):
    """
    A wrapper around the GPTModel that adds a classification head.
    """
    def __init__(self, gpt_model, num_classes):
        super().__init__()
        self.gpt = gpt_model
        # Get the dimensionality of the model from the gpt_model instance
        d_model = self.gpt.head.in_features
        # The classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Forward pass for the classification model.
        """
        # 1. Get the output from the base GPT model
        # The output `x` here has the shape (batch_size, seq_len, d_model)
        x = self.gpt.embedding(x)
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        for block in self.gpt.blocks:
            x = block(x, mask)
        x = self.gpt.ln_f(x)
        
        # 2. We only care about the output of the very last token in the sequence
        last_token_output = x[:, -1, :] # Shape: (batch_size, d_model)
        
        # 3. Pass this final token's output through the classification head
        logits = self.classifier(last_token_output) # Shape: (batch_size, num_classes)
        
        return logits