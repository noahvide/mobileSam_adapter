import torch
import torch.nn as nn
from models.mobileSam.mobile_sam.modeling.image_encoder import ImageEncoderViT
from .prompt_generator import PromptGenerator

class ImageEncoderViTAdapter(ImageEncoderViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_factor = 32
        self.prompt_type = 'highpass'
        self.tuning_stage = 1234
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        self.embed_dim = kwargs["embed_dim"]

        self.prompt_generator = PromptGenerator(
            self.scale_factor,
            self.prompt_type,
            self.embed_dim,
            self.tuning_stage,
            len(self.blocks),
            self.input_type,
            self.freq_nums,
            self.handcrafted_tune,
            self.embedding_tune,
            self.adaptor,
            self.img_size,
            self.patch_embed.proj.kernel_size[0],
        )

    def forward(self, x):
        inp = x
        x = self.patch_embed(x)
        embedding_feature = self.prompt_generator.init_embeddings(x)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, H, W = x.shape[:3]
        for i, blk in enumerate(self.blocks):
            x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x