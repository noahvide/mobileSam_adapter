import torch
import torch.nn as nn

from models.mobileSam.mobile_sam.modeling import Sam, TinyViT, PromptEncoder
from models.mobileSam.mobile_sam.modeling import TwoWayTransformer
from models.mobileSam.mobile_sam.modeling.mask_decoder import MaskDecoder
from adapters.mobilesam_adapter.prompt_generator import PromptGenerator
from adapters.lora.lora_module import apply_lora  # assuming you have this

class MobileSAMBase(Sam):
    def __init__(self,
                 img_size=1024,
                 pretrained_checkpoint=None):
        
        super().__init__(image_encoder=None, prompt_encoder=None, mask_decoder=None)

        # ------------------------------
        # 1. Initialize TinyViT (matches checkpoint)
        # ------------------------------
        self.image_encoder = TinyViT(
            img_size=img_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        # ------------------------------
        # 2. PromptEncoder and MaskDecoder
        # ------------------------------
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            input_image_size=img_size,
            image_embedding_size=(img_size // 16, img_size // 16),
            mask_in_chans=16,  # matches checkpoint
        )

        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )

        self.mask_decoder = MaskDecoder(
            transformer=transformer,
            transformer_dim=256,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.image_encoder = self.image_encoder
        self.prompt_encoder = self.prompt_encoder
        self.mask_decoder = self.mask_decoder
        
        # ------------------------------
        # 3. Load pretrained checkpoint
        # ------------------------------
        self.pretrained_param_count = 0
        if pretrained_checkpoint is not None:
            self.pretrained_param_count = self.load_pretrained_checkpoint(pretrained_checkpoint)

        # ------------------------------
        # 4. Freeze all pretrained weights
        # ------------------------------
        for name, param in self.named_parameters():
            param.requires_grad = False

        # ------------------------------
        # 5. Initialize adapter placeholders
        # ------------------------------
        self.adapter = None
        self.lora_applied = False

        # ------------------------------
        # 6. Logging
        # ------------------------------
        self._log_params()

    # ------------------------------------------------------------------
    # Add SAM-Adapter
    # ------------------------------------------------------------------
    def add_adapter(self, adapter_class=PromptGenerator, **kwargs):
        """Attach a SAM-Adapter prompt to the pretrained image encoder."""
        self.adapter = adapter_class(**kwargs)

        # Modify forward_features to inject adapter
        old_forward = self.image_encoder.forward_features

        def new_forward(x):
            features = old_forward(x)
            prompt = self.adapter.get_prompt(x, features)
            return features + prompt

        self.image_encoder.forward_features = new_forward

        # Make adapter parameters trainable
        for name, param in self.named_parameters():
            if hasattr(self, 'adapter') and name.startswith('adapter'):
                param.requires_grad = True

        print("[INFO] Adapter attached. Adapter parameters are trainable.")
        self._log_params()

    # ------------------------------------------------------------------
    # Add LoRA
    # ------------------------------------------------------------------
    def add_lora(self, **kwargs):
        """Apply LoRA to TinyViT layers."""
        self.image_encoder = apply_lora(self.image_encoder, **kwargs)
        self.lora_applied = True

        # Make LoRA parameters trainable
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        print("[INFO] LoRA applied. LoRA parameters are trainable.")
        self._log_params()

    # ------------------------------------------------------------------
    # Utility: log param counts
    # ------------------------------------------------------------------
    def _log_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Trainable params: {trainable_params/1e6:.3f}M")
        print(f"[INFO] Pretrained params: {self.pretrained_param_count/1e6:.3f}M")
        print(f"[INFO] Total params: {total_params/1e6:.3f}M")
        diff = total_params - (self.pretrained_param_count + trainable_params)
        if abs(diff) > 1e4:
            print(f"[WARN] Some layers ({diff/1e6:.3f}M params) not covered by checkpoint — likely adapter or LoRA layers.")
        else:
            print("[INFO] Parameter count check passed.")

    # ------------------------------------------------------------------
    # Load checkpoint (robust)
    # ------------------------------------------------------------------
    def load_pretrained_checkpoint(self, checkpoint_path: str) -> int:
        print(f"[INFO] Loading pretrained checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        pretrained_count = 0

        def safe_load(module, prefix):
            nonlocal pretrained_count
            subdict = {k.replace(f"{prefix}.", ""): v for k, v in checkpoint.items() if k.startswith(prefix)}
            if not subdict:
                print(f"[WARN] No weights found for {prefix}")
                return
            state_dict = module.state_dict()
            compatible = {k: v for k, v in subdict.items() if k in state_dict and state_dict[k].shape == v.shape}
            if compatible:
                module.load_state_dict(compatible, strict=False)
                loaded = sum(v.numel() for v in compatible.values())
                pretrained_count += loaded
                total_mod = sum(p.numel() for p in module.parameters())
                print(f"[INFO] {prefix:<15} loaded={loaded/1e6:>6.3f}M  total={total_mod/1e6:>6.3f}M")
            else:
                print(f"[WARN] No compatible weights loaded for {prefix}")

        # Load each submodule
        safe_load(self.image_encoder, "image_encoder")
        safe_load(self.prompt_encoder, "prompt_encoder")
        safe_load(self.mask_decoder, "mask_decoder")

        print(f"[INFO] Total pretrained params loaded: {pretrained_count/1e6:.3f}M")
        return pretrained_count