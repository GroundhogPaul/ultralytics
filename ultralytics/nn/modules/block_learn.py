class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):   # typical value for dim = 128
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads  # 8
        self.head_dim = dim // num_heads    # 16
        self.key_dim = int(self.head_dim * attn_ratio)  # 8
        self.scale = self.key_dim**-0.5 # 0.3535
        nh_kd = self.key_dim * num_heads # 64
        h = dim + nh_kd * 2 # 256
        self.qkv = Conv(dim, h, 1, act=False)   # Conv(128, 256, 1)
        self.proj = Conv(dim, dim, 1, act=False)    # Conv(128, 128, 1)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False) # Conv(128, 128, 3, 1, group = 128)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape # 1, 128 (must = dim), 160, 160
        N = H * W # 160 * 160 = 25600
        qkv = self.qkv(x) # 1, 256, 160, 160
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        ) # 按把很多channel分组，分成了qkv
        # qkv.view(1, 8, 32, 160*160)
        # split(8, 8, 16)
        # q: (1, 8, 8, 160*160)
        # k: (1, 8, 8, 160*160)
        # v: (1, 8, 16, 160*160)

        attn = (q.transpose(-2, -1) @ k) * self.scale # 
        # q.transpose(-2, -1): 1, 8, 160*160, 8
        # q.transpose(-2, -1) @ k : 1, 8, 160*160, 160*160
        attn = attn.softmax(dim=-1) # 1, 8, 160*160, 160*160
        
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        # attn.transpose(-2, -1): 1, 8, 160*160(softmaxed), 160*160
        # v @ attn.transpose(-2, -1): 1, 8, 16, 160*160
        # (v @ attn.transpose(-2, -1)).view(B, C, H, W): 1, 128, 160, 160
        
        x = self.proj(x)
        return x