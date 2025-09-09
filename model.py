import torch
from mini_torch import MyLayerNorm, MyLinear, MyEmbedding, MyModule, SinusoidalPositionalEncoding, TransformerBlock

"""
代码在做什么
1. 初始化 (__init__)
    self.token_emb：把输入 token id 映射到 d_model 维向量。
    self.pos_emb：给序列加上位置编码。
    self.blocks：构造了 num_layers 个 TransformerBlock，并通过 setattr 注册成子模块。
    self.ln_f：最终 LayerNorm。
    self.fc_out：最后的全连接层（Fully Connected Layer），把 hidden 向量投影到词表大小的 logits。

2. forward
    输入 (B,T) → 经过 embedding 和位置编码 → (B,T,d_model)。
    构造 causal mask（下三角），保证解码器只能看见过去。
    如果传了 padding_idx，再结合 padding mask → 同时屏蔽未来和 padding。
    循环调用每个 TransformerBlock，每层做：x=x+MHA(LN(x));x=x+FFN(LN(x))
    最终 LayerNorm + 输出层，得到 (B,T,vocab_size) 的 logits
"""
class DecoderOnlyTransformer(MyModule):
    def __init__(self, vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=128, max_len=128):
        super().__init__()
         # token embedding
        self.token_emb = MyEmbedding(vocab_size, d_model)  # token 索引（整数）映射成 d_model 维向量
        
        # 位置编码
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len)  # 位置编码
        
        # 手动注册子块, 堆叠 N 个 Block
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        for i, block in enumerate(self.blocks):
            setattr(self, f"block{i}", block)  # MyModule 风格注册子模块
        
        # 最终 LayerNorm
        self.ln_f = MyLayerNorm(d_model)   # 最终输出前的归一化层
        
        # 输出层 fc-fully Connected Layer
        self.fc_out = MyLinear(d_model, vocab_size) #把隐藏向量投影回 词表大小的 logits

    def forward(self, x, padding_idx=None):
        """
        x: (B, T) int64 token ids
        """
        assert x.dtype == torch.long, "输入 token 索引应为 torch.long"

        # 取 batch 和序列长度
        B, T = x.shape
        device = x.device

        # 词向量 + 位置编码
        h = self.token_emb(x)
        h = self.pos_emb(h)

        # 构造因果（causal）mask
        # causal mask: (B,1,T,T)
        causal = self.build_causal_mask(B, T, device)

        # 如果需要同时考虑 padding，可叠乘一个 padding mask：
        if padding_idx is not None:
            # (B,T) → (B,1,1,T) 广播到注意力维度
            valid = (x != padding_idx).view(B, 1, 1, T)
            attn_mask = causal & valid # 同时满足因果与非padding
        else:
            attn_mask = causal

        # 堆叠 TransformerBlock
        # 每个 block 内部是 Pre-Norm：x = x + MHA(LN(x))，再 x = x + FFN(LN(x))
        # 传进去的 mask 会在注意力里作为 attn_scores.masked_fill(~mask, -inf) 使用
        for i in range(len(self.blocks)):
            block = getattr(self, f"block{i}")
            h = block(h, mask=attn_mask)

        # 最终 LayerNorm 与输出投影
        h = self.ln_f(h)
        logits = self.fc_out(h) # (B,T,vocab_size)
        return logits

    def build_causal_mask(self, B, T, device):
        m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return m.view(1, 1, T, T).expand(B, 1, T, T)
