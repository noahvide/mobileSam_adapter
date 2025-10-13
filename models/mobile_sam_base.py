from .mobileSam.mobile_sam.modeling import Sam, ImageEncoderViT, TinyViT, PromptEncoder, MaskDecoder
from models.mobileSam.mobile_sam.modeling import TwoWayTransformer
from models.mobileSam.mobile_sam.modeling.mask_decoder import MaskDecoder

from adapters.mobilesam_adapter.image_encoder_with_adapter import ImageEncoderViTAdapter
from adapters.lora.lora_module import apply_lora

class MobileSAMBase(Sam):
    def __init__(self, img_size=224, embed_dim=768, use_adapter=False, use_lora=False):
        if use_adapter:
            image_encoder = ImageEncoderViTAdapter(img_size=img_size, embed_dim=embed_dim)
        elif use_lora:
            image_encoder = ImageEncoderViT(img_size=img_size, embed_dim=embed_dim)
            image_encoder = apply_lora(image_encoder)
        else:
            image_encoder = TinyViT(img_size=img_size)

        prompt_encoder = PromptEncoder(
            embed_dim=256,
            input_image_size=img_size,
            image_embedding_size=(img_size // 16, img_size // 16),
            mask_in_chans=1
        )

        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8
        )

        mask_decoder = MaskDecoder(
            transformer=transformer,
            transformer_dim=256,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        super().__init__(image_encoder, prompt_encoder, mask_decoder)


        for name, param in self.named_parameters():
            param.requires_grad = False  # freeze everything

        # If using adapter: unfreeze only the adapter layers
        if use_adapter:
            for name, param in self.named_parameters():
                if "prompt_generator" in name or "adapter" in name:
                    param.requires_grad = True

        # If using LoRA: unfreeze LoRA weights
        elif use_lora:
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        # Optional sanity check
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Trainable params: {trainable/1e6:.3f}M / {total/1e6:.3f}M total")