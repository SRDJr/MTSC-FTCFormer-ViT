import torch
import torch.nn as nn
from .tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM

class TCFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=10, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 k=5, sample_ratios=[0.25, 0.25, 0.25], 
                 patch_size=16, stride=16, 
                 beta=1.0,
                 use_fuzzy=False, use_softmax=True, use_cross_attention=False): 
        super().__init__()
        self.num_classes = num_classes
        self.depths = [3, 4, 6, 3]
        
        # Stage 1: Patch Embedding
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            stride=stride, 
            in_chans=in_chans, 
            embed_dim=embed_dims[0]
        )
        
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0]) for _ in range(self.depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stages 2-4: FTC (Fuzzy Token Clustering)
        for i in range(1, 4):
            # Pass 'beta' to the CTM module
            ctm = CTM(
                sample_ratio=sample_ratios[i-1], 
                embed_dim=embed_dims[i-1], 
                dim_out=embed_dims[i], 
                k=k,
                beta_init=beta, # <--- INJECT BETA HERE
                use_cross_attention=use_cross_attention, # <--- PASS IT DOWN
                use_softmax=use_softmax,                 # <--- PASS IT DOWN
                use_fuzzy=use_fuzzy                      # <--- PASS IT DOWN
            )
            
            block = nn.ModuleList([TCBlock(dim=embed_dims[i], num_heads=num_heads[i], 
                                           use_fuzzy=use_fuzzy, use_softmax=use_softmax)
                                   for _ in range(self.depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            
            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i+1}", block)
            setattr(self, f"norm{i+1}", norm)

        # Classification Head
        self.head = nn.Linear(embed_dims[3], num_classes)

    def forward(self, x):
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)

        # Prepare Dictionary
        B, N, _ = x.shape
        token_dict = {'x': x, 'map_size': [H, W], 
                      'idx_token': None, 'agg_weight': None}

        # Stages 2-4
        for i in range(1, 4):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i+1}")
            norm = getattr(self, f"norm{i+1}")

            token_dict = ctm(token_dict)
            for blk in block:
                token_dict = blk(token_dict)
            token_dict['x'] = norm(token_dict['x'])

        # Final Classification
        x = token_dict['x'].mean(dim=1)
        return self.head(x)



 