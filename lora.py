import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    A Low-Rank Adaptation (LoRA) layer that can be injected into any linear layer.
    """
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Create the low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # The original forward pass
        original_output = self.original_layer(x)
        
        # The LoRA path
        lora_output = (x @ self.lora_A) @ self.lora_B
        
        return original_output + lora_output