import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_utils import DropPath, to_2tuple, trunc_normal_
from .tcformer_utils import token2map, map2token

class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5, Cmerge=True, beta_init=1.0, use_cross_attention=False, use_softmax=True, use_fuzzy=True):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        self.k = k
        self.Cmerge = Cmerge

        # Normalize INPUT dimension
        self.norm = nn.LayerNorm(embed_dim)

        # Score using INPUT dimension
        if Cmerge:
            self.score = nn.Linear(embed_dim, embed_dim)
        else:
            self.score = nn.Linear(embed_dim, 1)

        # Project to Output Dimension AFTER merging
        self.proj = nn.Linear(embed_dim, dim_out)

        # --- INNOVATION I: Adaptive Density Parameter ---
        # Initialize with the dataset-specific beta value
        self.beta = nn.Parameter(torch.tensor([beta_init]), requires_grad=True)

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            # Initialize our new Fuzzy Cross Attention class here!
            # It operates on embed_dim before the final projection
            self.cross_attn = LearnableGaussianCrossAttention(dim=embed_dim, num_heads=8, use_softmax=use_softmax, use_fuzzy=use_fuzzy)

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        
        # --- SAVE ORIGINAL TOKENS HERE ---
        # We need these untouched for Cross-Attention later
        x_orig = token_dict['x'] 
        # ---------------------------------
        
        # 1. Normalize Input
        x = self.norm(x_orig)
        B, N, C = x.shape

        # 2. Calculate Token Scores
        token_score = self.score(x)
        max_scores = token_score.max(dim=1, keepdim=True).values
        token_weight = (token_score - max_scores).exp()
        
        token_dict['x'] = x
        token_dict['token_score'] = token_score.mean(-1, keepdim=True) if self.Cmerge else token_score

        # 3. Calculate target cluster count
        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        
        # 4. Cluster (DPC-FKNN) with Adaptive Beta
        idx_cluster, cluster_num = self.cluster_dpc_fknn_wsn(token_dict, cluster_num, self.k)
        
        # 5. Merge Tokens
        down_dict = self.Cmerge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        
        """
        ======================================================================
        STEP 6: TOKEN INTERACTION (Cross-Attention)
        We do this BEFORE the final projection so that the Merged Tokens (Q) 
        and the Original Tokens (K, V) have the exact same channel dimensions!
        ======================================================================
        """
        if getattr(self, 'use_cross_attention', False):
            q_merged = down_dict['x']
            
            # Fuzzy Cross-Attention between Merged (Q) and Original (K, V)
            attn_out, _ = self.cross_attn(q_merged, x_orig, token_score)
            
            # Add the recovered details via residual connection
            down_dict['x'] = down_dict['x'] + attn_out
        # ====================================================================

        # 7. Project Features 
        # Now we can safely project to the new dimension for the next Transformer Stage
        down_dict['x'] = self.proj(down_dict['x'])

        # Update map size approximation
        H, W = token_dict['map_size']
        H = math.ceil(H * math.sqrt(self.sample_ratio))
        W = math.ceil(W * math.sqrt(self.sample_ratio))
        down_dict['map_size'] = [H, W]

        return down_dict

    def cluster_dpc_fknn_wsn(self, token_dict, cluster_num, k=5):
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            
            # Distance Matrix
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            # KNN
            dist_nearest, nn_indices = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            
            # --- INNOVATION I: Adaptive Beta Application ---
            # Clamp to prevent math errors (negative/zero)
            adaptive_beta = self.beta.clamp(min=0.01)

            """
            ======================================================================
            OPTION A: ORIGINAL PAPER MATH + OUR ADAPTIVE BETA
            The paper uses a rigid fractional penalty and a global context term.
            We have enhanced their exact math by injecting our adaptive beta into 
            the exponential numerator. 
            
            TO USE: Uncomment the block below, and comment out OPTION B.
            
            # 1. KNN Term (mu for nearest neighbors)
            # num_knn = torch.exp(-(adaptive_beta * dist_nearest ** 2))
            # den_knn = dist_nearest + 1.0
            # mu_knn = num_knn / den_knn
            # term1 = mu_knn.mean(dim=-1)
            
            # 2. Global Context Term (mu for all N tokens)
            # num_all = torch.exp(-(adaptive_beta * dist_matrix ** 2))
            # den_all = dist_matrix + 1.0
            # mu_all = num_all / den_all
            # term2 = mu_all.mean(dim=-1)
            
            # density = term1 + term2
            ======================================================================
            """

            # ======================================================================
            # OPTION B: OUR FULL INNOVATION (Currently Active)
            # We average the squared distances FIRST before applying the exponential.
            # This drastically improves mathematical stability on chaotic time-series.
            density = (-(adaptive_beta * dist_nearest ** 2)).mean(dim=-1).exp()
            # ======================================================================

            # Noise to break ties
            density = density + torch.rand_like(density) * 1e-6


            # Distance Indicator (Delta)
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # Center Selection
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Assignment via WSN
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]

            """
            ======================================================================
            ORIGINAL PAPER EQUATION 4, 5, 6: Spatial Connectivity Score (SCS)
            The paper uses SCS for token assignment, requiring exact SNN intersections.
            If you implement a 'calculate_SCS' function from the original GitHub, 
            UNCOMMENT the line below AND comment out our WSN line.
            
            # SCS_score = self.calculate_SCS(dist_selected, dist_matrix, nn_indices, index_down)
            ======================================================================
            """
            
            # --- OUR IMPLEMENTATION: Weighted Shared Neighbor (WSN) ---
            # --- COMMENT THIS LINE OUT IF USING ORIGINAL SCS EQUATION ---
            WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            # -------------------------------------------------------------------
            
            # Final Assignment Score (Change WSN_score to SCS_score if swapped)
            final_score = self.index_points(dist_matrix, index_down) - (100 * WSN_score)
            idx_cluster = final_score.argmin(dim=1)

            # Force centers to map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
            
        return idx_cluster, cluster_num

    def calculate_WSN(self, dist_selected, dist_matrix, nn_indices, index_down):
        W = (1 / (dist_selected + 1)).sum(dim=-1)
        W = W.unsqueeze(-1) + W.unsqueeze(-2) 
        
        knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)
        snn = knn_mask @ knn_mask.transpose(1, 2)

        WSN = snn * W
        WSN = self.index_points(WSN, index_down)
        return WSN

    def calculate_SCS(self, dist_selected, dist_matrix, nn_indices, index_down):
        """
        ORIGINAL PAPER EQUATION 4, 5, 6: Spatial Connectivity Score (SCS)
        Calculates exact Shared Nearest Neighbors (SNN) and Closeness to Neighbors (CN).
        """
        # Equation 5: Closeness to Neighbors (CN)
        W = (1 / (dist_selected + 1)).sum(dim=-1) 
        CN = W.unsqueeze(-1) + W.unsqueeze(-2)  # CN(i,j) = W_i + W_j
        
        # Equation 4: Shared Nearest Neighbors (SNN) exact intersection
        knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)
        SNN = knn_mask @ knn_mask.transpose(1, 2)

        # Equation 6: SCS = CN * |SNN|
        SCS = SNN * CN
        SCS = self.index_points(SCS, index_down)
        
        return SCS
    
    def Cmerge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight):
        x = token_dict['x']
        B, N, C = x.shape

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num

        all_weight = token_weight.new_zeros(B * cluster_num, C)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, C))
        all_weight = all_weight + 1e-6 
        
        norm_weight = token_weight / all_weight[idx]

        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        return {'x': x_merged, 'token_num': cluster_num, 'agg_weight': None, 'idx_token': None}

    def index_points(self, points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        return points[batch_indices, idx, :]

# --- Standard Blocks ---

class LearnableGaussianAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, use_softmax=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # This is our Phase 1 vs Phase 2 toggle!
        self.use_softmax = use_softmax

        # Standard QKV linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # --- THE FUZZY PARAMETERS ---
        # Each head gets its own learnable mu (target) and sigma (strictness)
        self.mu = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.sigma = nn.Parameter(torch.ones(num_heads, 1, 1) * 2.0)

    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Raw Similarity Scores
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # --- THE FUZZY GATE ---
        # Calculate Gaussian Membership: exp( -(x - mu)^2 / 2*sigma^2 )
        fuzzy_scores = torch.exp(-((scores - self.mu)**2) / (2 * (self.sigma + 1e-8)**2))

        # --- PHASE 1 vs PHASE 2 LOGIC ---
        if self.use_softmax:
            # Phase 1: FANTF Style (Safe, Stable, Normalizes the fuzzy score)
            attn_weights = fuzzy_scores.softmax(dim=-1)
        else:
            # Phase 2: Softmax-Free (Independent Gating)
            attn_weights = fuzzy_scores

        # Context Aggregation
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        # Return both output AND weights for interpretability
        return out, attn_weights
    
class LearnableGaussianCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, use_softmax=True, use_fuzzy=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.use_softmax = use_softmax
        self.use_fuzzy=use_fuzzy

        # Separate projections for Q (merged tokens) and K, V (original tokens)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        # The Fuzzy Parameters
        self.mu = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.sigma = nn.Parameter(torch.ones(num_heads, 1, 1) * 2.0)

    def forward(self, q_merged, kv_orig, channel_scores):
        B, N_m, C = q_merged.shape
        _, N_o, _ = kv_orig.shape

        q = self.q_proj(q_merged).reshape(B, N_m, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(kv_orig).reshape(B, N_o, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(kv_orig).reshape(B, N_o, self.num_heads, C // self.num_heads).transpose(1, 2)

        # 1. Paper's Exact Score Calculation (Qm * Ko^T)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add the channel modulator (Eq 8)
        avg_P = channel_scores.mean(dim=-1).unsqueeze(1).unsqueeze(1)
        scores = scores + avg_P

        # --- THE FIX: Respect the master fuzzy toggle ---
        if self.use_fuzzy:
            # 2. OUR Fuzzy Gate 
            fuzzy_scores = torch.exp(-((scores - self.mu)**2) / (2 * (self.sigma + 1e-8)**2))

            # 3. Phase 1 vs Phase 2 Logic
            if self.use_softmax:
                attn_weights = fuzzy_scores.softmax(dim=-1)
            else:
                attn_weights = fuzzy_scores
        else:
            # PURE BASELINE (Crisp Logic): Standard Softmax, no fuzzy gate
            attn_weights = scores.softmax(dim=-1)
        # ------------------------------------------------

        # 4. Context Aggregation with Original Values (Vo)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N_m, C)
        return self.out_proj(out), attn_weights
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), act_layer(), nn.Linear(int(dim*mlp_ratio), dim))
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

"""
======================================================================
ORIGINAL PAPER OMISSION 1: Convolutional FFN (CPEG)
The paper uses a Feed-Forward Network with a Depth-wise Convolution 
to leak 2D spatial information into the tokens. We use a standard MLP 
instead because precise 2D spatial coordinates are less critical for 
time-series representations.

To use this, uncomment the class below and replace `self.mlp` in the 
TCBlock with `self.mlp = ConvMlp(dim, int(dim*mlp_ratio))`
======================================================================
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # The 2D Depth-wise Convolution
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # NOTE: If used after FTCM, x must be scattered back to a 2D map 
        # using `token2map` before applying this convolution!
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
======================================================================
"""

class TCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, if_CTM=True,
                 use_fuzzy=False, use_softmax=True): 
        super().__init__()
        self.use_fuzzy = use_fuzzy
        self.norm1 = norm_layer(dim)

        """
        ======================================================================
        ORIGINAL PAPER OMISSION 2: Spatial Reduction (SR) Layer
        The paper uses a strided CNN to reduce the sequence length of Keys 
        and Values to save GPU VRAM on massive natural images. 
        Uncomment the lines below to initialize the SR convolutions.
        
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm_sr = norm_layer(dim)
        ======================================================================
        """
        
        # --- THE DROP-IN REPLACEMENT SWITCH ---
        if self.use_fuzzy:
            self.attn = LearnableGaussianAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_softmax=use_softmax)
        else:
            # Fall back to standard PyTorch attention for your baseline
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
            
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), act_layer(), nn.Linear(int(dim*mlp_ratio), dim))
    
    def forward(self, token_dict):
        x = token_dict['x']
        
        """
        ======================================================================
        ORIGINAL PAPER OMISSION 2: SR Layer Forward Pass
        To apply the SR layer, the 1D sequence must be temporarily folded 
        back into a 2D image, convolved, and flattened. 
        Uncomment the logic below to generate the reduced `x_kv` tensor.
        
        # B, N, C = x.shape
        # H, W = token_dict['map_size']
        # if getattr(self, 'sr_ratio', 1) > 1:
        #     # NOTE: Requires `token2map` if used after FTCM merging!
        #     x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        #     x_kv = self.sr(x_2d).reshape(B, C, -1).transpose(1, 2)
        #     x_kv = self.norm_sr(x_kv)
        # else:
        #     x_kv = x
        ======================================================================
        """
        
        if self.use_fuzzy:
            # Custom Phase 1/2 Innovation: Fuzzy Self-Attention
            attn_out, fuzzy_weights = self.attn(self.norm1(x))
            
            """
            # IF USING SR LAYER: You must modify `LearnableGaussianAttention` 
            # to accept a separate `kv` argument, then uncomment the line below:
            # attn_out, fuzzy_weights = self.attn(self.norm1(x), x_kv)
            """
            
            self.last_fuzzy_weights = fuzzy_weights.detach() 
            x = x + attn_out
        else:
            # Baseline: Standard PyTorch Attention 
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            
            """
            # IF USING SR LAYER: Uncomment the line below to pass the reduced 
            # Keys and Values to the standard PyTorch attention mechanism:
            # x = x + self.attn(self.norm1(x), x_kv, x_kv)[0]
            """
            
        """
        ======================================================================
        ORIGINAL PAPER OMISSION 1: Convolutional MLP
        The paper uses a Conv2d in the MLP for implicit positional encoding.
        Uncomment the line below (and comment the standard mlp) to use it.
        
        # H, W = token_dict['map_size']
        # x = x + self.mlp(self.norm2(x), H, W) 
        ======================================================================
        """
        # Standard MLP (Currently Active)
        x = x + self.mlp(self.norm2(x))
        
        token_dict['x'] = x
        return token_dict
