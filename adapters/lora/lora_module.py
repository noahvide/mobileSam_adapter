import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.
    """
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA A and B projection matrices
        self.lora_A = nn.Parameter(torch.zeros((r, base_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x):
        result = self.base(x)
        lora_update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_update
    

def apply_lora(model, target_keywords=("attn", "qkv", "proj"), r=8, alpha=16, dropout=0.05):
    """
    Recursively replace Linear layers whose name matches `target_keywords` with LoRALinear.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora(module, target_keywords, r=r, alpha=alpha, dropout=dropout)
    return model