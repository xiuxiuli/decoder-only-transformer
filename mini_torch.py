import math
import torch
import torch.nn as nn

# ========= 核心：简化版 Module / Parameter 收集机制 =========
# 让一个 Tensor 变成“模型参数”
class MyParameter(torch.Tensor):
    """
    简化版 Parameter:
    - 就是一个 torch.Tensor
    - 默认 requires_grad=True
    - 自动注册到 nn.Module.parameters()
    """
    def __new__(cls, data, requires_grad=True):
        # 保证 data 先转换成 Tensor
        # isinstance(obj, cls) 检查对象 obj 是不是 cls 类或其子类的实例
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        # 用 Tensor 的 __new__ 构建
        return torch.Tensor._make_subclass(cls, data, requires_grad)

class MyModule:
    """
    超简化版 nn.Module：
      - 自动注册子模块和参数（通过 __setattr__）
      - 提供 parameters()/named_parameters() 递归收集
      - 提供 __call__ -> forward 调用约定
      - train()/eval() 开关（示范用）
    """
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        # 自动注册 参数 / 子模块
        if isinstance(value, MyParameter):
            self._parameters[name] = value
        elif isinstance(value, MyModule):
            self._modules[name] = value

        # 其他普通属性
        object.__setattr__(self, name, value)
    
    # --- 递归参数收集 ---
    def parameters(self):
        # 先返回当前模块的参数
        for p in self._parameters.values():
            yield p
        # 再递归子模块
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=""):
        # 1. 遍历当前模块的参数
        for n, p in self._parameters.items():
            yield (f"{prefix}{n}", p)

        # 2. 递归遍历子模块的参数
        for child_name, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}{child_name}.")

    # --- 统一调用入口：model(x) -> forward(x) ---
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self, mode: bool = True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def to(self, device):
        # 把所有参数和子模块递归搬到 device 上
        for name, p in self._parameters.items():
            self._parameters[name] = p.to(device)
        for m in self._modules.values():
            m.to(device)
        return self
    
# ===== Embedding =====
class MyEmbedding(MyModule):
    def __init__(self, vocab_size, d_model, padding_idx=None, init_std=0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        w = torch.randn(vocab_size, d_model) * init_std
        if padding_idx is not None:
            w[padding_idx].zero_()
        self.weight = MyParameter(w)

    def forward(self, x):
        out = self.weight[x]                      # (B, T, d_model)
        if self.padding_idx is not None:
            out = out.clone()
            out[x == self.padding_idx] = 0.0
        return out

    

# ========= 基础层：Linear / ReLU / Sequential / Loss =========
class MyLinear(MyModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 权重 W 通常用 kaiming/xavier 初始化。
        # Kaiming-uniform 初始化 何恺明给ReLU 激活函数找到合适的权重初始分布，避免梯度消失/爆炸
        # 提出了残差网络
        bound = 1.0 / math.sqrt(in_features)
        #(out_features 行, in_features 列)
        w = torch.empty(out_features, in_features).uniform_(-bound, bound)
        self.weight = MyParameter(w)

        # 偏置用均匀分布初始化（范围跟 fan-in 有关）
        if bias:
            b = torch.empty(out_features).uniform_(-bound, bound)
            self.bias = MyParameter(b)
        else:
            self.bias = None

    # Linear(x)=x*W+b
    def forward(self, x):
        # 保存输入，供 backward 用
        self.x = x
        # 矩阵乘法规则是 (N, in_features) @ (in_features, out_features) → (N, out_features)
        # y=X*W-TRASNPOS

        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
    
    def backward(self, dY): #dY: 损失函数L对本层输出y的梯度: dY=∂y/∂L​ 
        """
        dY: (N, out_features)，损失对输出的梯度
        """
        # 1. 权重梯度
        self.dW = dY.T @ self.x      # (out_features, in_features)

        # 2. 偏置梯度
        if self.bias is not None:
            self.db = dY.sum(dim=0)     # (out_features,)

        # 3. 输入梯度
        dx = dY @ self.weight   # (N, in_features)

        return dx
    
# ========== 位置编码 ==========
class SinusoidalPositionalEncoding(MyModule):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0) / d_model))

        # 在 Python 里，切片 [start:stop:step]
        # pe[:, 0::2]     : → 保留第一个维度（batch=1), 0::2 → 在最后一个维度（d_model 方向）取偶数索引。
        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数维

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.pe = pe

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        return x + self.pe[:, :x.size(1)]

# ========== 做归一化 ==========  
# 它在 最后一维 (d_model) 上做均值方差归一化
# B×T×d_model​ | B (Batch size), T (Sequence length, how many token), d_model (Embedding dimension)
class MyLayerNorm(MyModule):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        # # gamma, beta: 可学习参数，分别是缩放系数和偏置
        self.gamma = MyParameter(torch.ones(d_model))   # 初始化为 1
        self.beta = MyParameter(torch.zeros(d_model))   # 初始化为 0
        self.eps = eps  # 防止除零的小常数

    def forward(self, x):
        # x: (B, T, d_model)
        mean = x.mean(dim=-1, keepdim=True) # (B, T, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)   # (B, T, 1)
        x_hat = (x-mean) / torch.sqrt(var +self.eps)    # 标准化
        return self.gamma * x_hat + self.beta

# ========== 多头注意力机制 ==========
class MyMultiHeadAttention(MyModule):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0 #因为多头注意力机制是把整个矩阵在维度方向切开几份的
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V 投影
        self.Wq = MyLinear(d_model, d_model)
        self.Wk = MyLinear(d_model, d_model)
        self.Wv = MyLinear(d_model, d_model)

        # 输出投影
        self.Wo = MyLinear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        H, D = self.num_heads, self.d_head

        # 新形状：(B, T, H, D) | B = batch, T = 序列长度, H = 注意力头数, D = 每个头的维度
        q = self.Wq(x).view(B, T, H, D)
        k = self.Wk(x).view(B, T, H, D)
        v = self.Wv(x).view(B, T, H, D)

        # 转置成 (B, H, T, D) 方便做注意力
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 注意力分数 (B, H, T, T)    (q @ kᵀ)/√D → 计算相似度分数
        attn_scores = (q @ k.transpose(-2, -1)) /math.sqrt(D)  # 相关性, 除以 √D 缩放
        if mask is not None:
            # 在 decoder-only 里，mask屏蔽掉未来token/padding 用来防止看到未来的token. mask == 0 的地方用负无穷替换
            # 允许 mask 为 (B,1,T,T) 或 (B,H,T,T)；自动广播
            # 约定：mask==1 表示允许，mask==0 表示屏蔽
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        
        attn = torch.softmax(attn_scores, dim=-1) #负无穷在softmax 的输出概率就是 0

        # 注意力加权 (B, H, T, D)   把分数转成概率，表示注意力分布
        out = attn @ v  # (B,H,T,D)

        # 合并头 (B, T, C)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)    # (B,T,C)
        return self.Wo(out)

# ==========  ==========
class MyFeedForward(MyModule):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = MyLinear(d_model, d_ff)
        self.fc2 = MyLinear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
        
# ========== Transformer Block ==========
class TransformerBlock(MyModule):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = MyLayerNorm(d_model) # ln1 专门服务于 注意力子层
        self.attn = MyMultiHeadAttention(d_model, num_heads)
        self.ln2 = MyLayerNorm(d_model) # ln2 专门服务于 前馈网络子层
        self.ff = MyFeedForward(d_model, d_ff)

    # Pre-Norm
    # 先对 x 做 ln1(x)，再送进 attn，最后残差回来 → Pre-Norm
    # 再对 x 做 ln2(x)，送进 FFN，最后残差回来 → Pre-Norm
    def forward(self, x, mask=None):
        # Pre-Norm + 残差
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x