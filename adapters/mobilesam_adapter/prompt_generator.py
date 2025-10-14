# prompt_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PromptGenerator(nn.Module):
    def __init__(self, embed_dim=256, scale_factor=4, freq_nums=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.freq_nums = freq_nums

        # Project handcrafted FFT (3-ch) -> embed_dim
        self.handcrafted_proj = nn.Conv2d(3, self.embed_dim, kernel_size=1, bias=True)

        # Modules following SAM-Adapter’s design
        # Note: these expect last-dim == embed_dim
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim // scale_factor)
        self.shared_mlp = nn.Linear(self.embed_dim // scale_factor, self.embed_dim)
        self.lightweight_mlp = nn.Sequential(
            nn.Linear(self.embed_dim // scale_factor, self.embed_dim // scale_factor),
            nn.GELU()
        )

        self._init_weights()

    def _init_weights(self):
        # initialize linear weights and conv
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # simple conv init
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fft(self, x, rate=0.25):
        """High-pass frequency extraction like SAM-Adapter.
           Input x: [B, 3, H, W]. Output: same shape [B, 3, H, W].
        """
        B, C, H, W = x.shape
        mask = torch.zeros_like(x)
        line = int((H * W * rate) ** 0.5 // 2)
        # make sure line >= 1
        line = max(1, line)
        mask[:, :, H//2-line:H//2+line, W//2-line:W//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        inv = torch.fft.ifft2(torch.fft.ifftshift(fft), norm="forward").real
        return torch.abs(inv)

    def get_prompt(self, x, embedding_feature):
        """
        Args:
            x: original input image [B, 3, H, W]
            embedding_feature: TinyViT output [B, C, H', W']
        Returns:
            prompt: same shape as embedding_feature [B, C, H', W']
        """
        B, C, H, W = embedding_feature.shape
        # handcrafted frequency-based feature, resized to match encoder output spatial size
        handcrafted = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        handcrafted = self.fft(handcrafted, self.freq_nums)   # [B, 3, H, W]

        # project handcrafted to embed_dim channels
        handcrafted_proj = self.handcrafted_proj(handcrafted)  # [B, C, H, W]

        # flatten both to shape [B, H*W, C]
        ef_flat = embedding_feature.permute(0, 2, 3, 1).reshape(B, H*W, C)
        hf_flat = handcrafted_proj.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # project to low-dim, fuse and project back (SAM-Adapter style)
        ef_low = self.embedding_generator(ef_flat)   # [B, HW, C//scale]
        hf_low = self.embedding_generator(hf_flat)   # [B, HW, C//scale]

        fused = self.lightweight_mlp(hf_low + ef_low)   # [B, HW, C//scale]
        prompt_flat = self.shared_mlp(fused)            # [B, HW, C]

        prompt = prompt_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return prompt


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from itertools import repeat

# def to_2tuple(x):
#     if isinstance(x, (list, tuple)):
#         return x
#     return tuple(repeat(x, 2))

# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#     l = norm_cdf((a - mean) / std)
#     u = norm_cdf((b - mean) / std)
#     with torch.no_grad():
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#     return tensor

# class PatchEmbed2(nn.Module):
#     """Image to Patch Embedding for prompts"""
#     def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=320):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         return x



# class PromptGenerator(nn.Module):
#     def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
#                  freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
#         """
#         Args:
#         """
#         super(PromptGenerator, self).__init__()
#         self.scale_factor = scale_factor
#         self.prompt_type = prompt_type
#         self.embed_dim = embed_dim
#         self.input_type = input_type
#         self.freq_nums = freq_nums
#         self.tuning_stage = tuning_stage
#         self.depth = depth
#         self.handcrafted_tune = handcrafted_tune
#         self.embedding_tune = embedding_tune
#         self.adaptor = adaptor

#         self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
#         self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
#         for i in range(self.depth):
#             lightweight_mlp = nn.Sequential(
#                 nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
#                 nn.GELU()
#             )
#             setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

#         self.prompt_generator = PatchEmbed2(img_size=img_size,
#                                                    patch_size=patch_size, in_chans=3,
#                                                    embed_dim=self.embed_dim//self.scale_factor)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def init_embeddings(self, x):
#         N, C, H, W = x.shape #permute(0, 3, 1, 2)
#         print(x.shape)
#         x = x.reshape(N, C, H*W).permute(0, 2, 1)
#         return self.embedding_generator(x)

#     def init_handcrafted(self, x):
#         x = self.fft(x, self.freq_nums)
#         print(x.shape)
#         return self.prompt_generator(x)

#     def get_prompt(self, handcrafted_feature, embedding_feature):
#         N, C, H, W = handcrafted_feature.shape
#         handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1)
#         print(f"{handcrafted_feature.shape = }")

#         prompts = []
#         for i in range(self.depth):
#             lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
#             # prompt = proj_prompt(prompt)
#             prompt = lightweight_mlp(handcrafted_feature + embedding_feature)
#             prompts.append(self.shared_mlp(prompt))
#         return prompts

#     def forward(self, x):
#         if self.input_type == 'laplacian':
#             pyr_A = self.lap_pyramid.pyramid_decom(img=x, num=self.freq_nums)
#             x = pyr_A[:-1]
#             laplacian = x[0]
#             for x_i in x[1:]:
#                 x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
#                 laplacian = torch.cat([laplacian, x_i], dim=1)
#             x = laplacian
#         elif self.input_type == 'fft':
#             x = self.fft(x, self.freq_nums)
#         elif self.input_type == 'all':
#             x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

#         # get prompting
#         prompt = self.prompt_generator(x)

#         if self.mode == 'input':
#             prompt = self.proj(prompt)
#             return prompt
#         elif self.mode == 'stack':
#             prompts = []
#             for i in range(self.depth):
#                 proj = getattr(self, 'proj_{}'.format(str(i)))
#                 prompts.append(proj(prompt))
#             return prompts
#         elif self.mode == 'hierarchical':
#             prompts = []
#             for i in range(self.depth):
#                 proj_prompt = getattr(self, 'proj_prompt_{}'.format(str(i)))
#                 prompt = proj_prompt(prompt)
#                 prompts.append(self.proj_token(prompt))
#             return prompts

#     def fft(self, x, rate):
#         # the smaller rate, the smoother; the larger rate, the darker
#         # rate = 4, 8, 16, 32
#         mask = torch.zeros(x.shape).to(x.device)
#         w, h = x.shape[-2:]
#         line = int((w * h * rate) ** .5 // 2)
#         mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

#         fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
#         # mask[fft.float() > self.freq_nums] = 1
#         # high pass: 1-mask, low pass: mask
#         fft = fft * (1 - mask)
#         # fft = fft * mask
#         fr = fft.real
#         fi = fft.imag

#         fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
#         inv = torch.fft.ifft2(fft_hires, norm="forward").real

#         inv = torch.abs(inv)

#         return inv

# # class PromptGenerator(nn.Module):
# #     def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
# #                  freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
# #         super().__init__()
# #         self.scale_factor = scale_factor
# #         self.prompt_type = prompt_type
# #         self.embed_dim = embed_dim
# #         self.input_type = input_type
# #         self.freq_nums = freq_nums
# #         self.tuning_stage = tuning_stage
# #         self.depth = depth
# #         self.handcrafted_tune = handcrafted_tune
# #         self.embedding_tune = embedding_tune
# #         self.adaptor = adaptor
        
# #         self.handcrafted_proj = nn.Linear(10, self.embed_dim // self.scale_factor)  # C_h=10 -> 8
# #         # self.handcrafted_proj = nn.Linear(10, self.embed_dim // self.scale_factor)  # e.g., 10 -> 32
# #         # self.handcrafted_proj = nn.Linear(10, self.embed_dim)
# #         # self.handcrafted_proj = nn.Linear(self.embed_dim, self.embed_dim)

# #         # shared MLP: input = embed_dim // scale_factor, output = embed_dim
# #         self.shared_mlp = nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)

# #         # embedding generator: input = embed_dim, output = embed_dim // scale_factor
# #         self.embedding_generator = nn.Linear(256, self.embed_dim  // self.scale_factor)

# #         # lightweight MLPs for each depth
# #         for i in range(self.depth):
# #             lightweight_mlp = nn.Sequential(
# #                 nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim // self.scale_factor),
# #                 nn.GELU()
# #             )
# #             setattr(self, f'lightweight_mlp_{i}', lightweight_mlp)

# #         # PatchEmbed for handcrafted features
# #         self.prompt_generator = PatchEmbed2(
# #             img_size=img_size,
# #             patch_size=patch_size,
# #             in_chans=3,
# #             embed_dim=self.embed_dim // self.scale_factor  # e.g., 32 if 256//8
# #         )

# #         self.apply(self._init_weights)

# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Linear):
# #             trunc_normal_(m.weight, std=.02)
# #             if m.bias is not None:
# #                 nn.init.constant_(m.bias, 0)
# #         elif isinstance(m, nn.LayerNorm):
# #             nn.init.constant_(m.bias, 0)
# #             nn.init.constant_(m.weight, 1.0)
# #         elif isinstance(m, nn.Conv2d):
# #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #             fan_out //= m.groups
# #             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
# #             if m.bias is not None:
# #                 m.bias.data.zero_()

# #     def init_embeddings(self, x):
# #         B, H_patch, W_patch, C = x.shape  # C = 256 here
# #         x = x.reshape(B, H_patch*W_patch, C)  # [B, N, C]
# #         return self.embedding_generator(x)      # [B, N, C//scale_factor]
    
# #     def init_handcrafted(self, x, target_size):
# #         # x: B C H W
# #         hf = self.fft(x, self.freq_nums)  # returns BCHW
# #         # resize to match block feature map
# #         hf = F.interpolate(hf, size=target_size, mode='bilinear', align_corners=False)
# #         return self.prompt_generator(hf)

# #     def get_prompt(self, handcrafted_feature, embedding_feature):
# #         # handcrafted_feature: B H W C -> flatten to B N C
# #         print(handcrafted_feature.shape)
# #         B, H, W, C_h = handcrafted_feature.shape
# #         handcrafted_flat = handcrafted_feature.reshape(B, H*W, C_h)
# #         handcrafted_proj_flat = self.handcrafted_proj(handcrafted_flat)  # [B, N, 8]

# #         combined = handcrafted_proj_flat + embedding_feature  # shapes match: [B, N, 8]

# #         prompts = []
# #         for i in range(self.depth):
# #             mlp = getattr(self, f'lightweight_mlp_{i}')
# #             prompt = mlp(combined)
# #             prompts.append(self.shared_mlp(prompt))
# #         return prompts

# #     def fft(self, x, rate):
# #         # x: B C H W
# #         B, C, H, W = x.shape
# #         mask = torch.zeros_like(x)
# #         line = int((H * W * rate) ** 0.5 // 2)
# #         mask[:, :, H//2-line:H//2+line, W//2-line:W//2+line] = 1
# #         fft = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
# #         fft = fft * (1 - mask)
# #         fft_hires = torch.fft.ifftshift(fft)
# #         inv = torch.fft.ifft2(fft_hires, norm='forward').real
# #         return torch.abs(inv)
    

# #     def forward(self, x):
# #         """
# #         x: B C H W
# #         returns: handcrafted feature map of shape B H_patch W_patch C_embed
# #         """
# #         # Get high-frequency (handcrafted) features
# #         handcrafted = self.fft(x, self.freq_nums)  # B C H W
# #         # Resize to patch embedding size
# #         # assume the patch size of your encoder: patch_H, patch_W
# #         patch_H = x.shape[2] // 16  # if patch_size=16
# #         patch_W = x.shape[3] // 16
# #         handcrafted = F.interpolate(handcrafted, size=(patch_H, patch_W),
# #                                     mode='bilinear', align_corners=False)

# #         # Convert to patch embedding
# #         patch_embed = self.prompt_generator(handcrafted)  # B C_embed H_patch W_patch
# #         # Permute to B H W C for transformer
# #         patch_embed = patch_embed.permute(0, 2, 3, 1)
# #         return patch_embed