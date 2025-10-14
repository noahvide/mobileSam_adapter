# image_encoder_with_adapter.py (TinyViTAdapter)
import torch
import torch.nn as nn
from models.mobileSam.mobile_sam.modeling import TinyViT
from .prompt_generator import PromptGenerator

class TinyViTAdapter(nn.Module):
    def __init__(self, img_size=1024, device='cpu'):
        super().__init__()
        self.image_encoder = TinyViT(
            img_size=img_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320], 
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
        )
        self.img_size = img_size

        # create prompt generator with the correct embed_dim
        self.prompt_generator = PromptGenerator(
            embed_dim=256,
            scale_factor=4,
            freq_nums=0.25,
        )

    def forward(self, x):
        ef = self.image_encoder.forward_features(x)  # [B, C, H, W]
        prompt = self.prompt_generator.get_prompt(x, ef)
        return ef + prompt  # Inject adapter prompt into final embedding
        # B, C, H, W = x.shape
        # print(f"{B, C, H, W = }")
        # # Patch embedding
        # x_patches = self.image_encoder.patch_embed(x)
        # print(f"{x_patches.shape = }")

        # # embedding feature
        # embedding_feature = self.prompt_generator.init_embeddings(x_patches)  # NO permute

        # # handcrafted feature
        # handcrafted_feature = self.prompt_generator(x)  # B H_patch W_patch C_embed_handcrafted

        # # combine
        # prompts = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)

        # if self.pos_embed is not None:
        #     x_patches = x_patches + self.pos_embed

        # outs = []
        # for i, blk in enumerate(self.blocks):
        #     B, H_blk, W_blk, C_blk = x_patches.shape
        #     x_patches = x_patches + prompts[i].reshape(B, H_blk, W_blk, -1)
        #     x_patches = blk(x_patches)
        #     if i in self.out_indices:
        #         outs.append(x_patches)

        # # final neck
        # x_out = self.neck(x_patches.permute(0,3,1,2))
        # return x_out