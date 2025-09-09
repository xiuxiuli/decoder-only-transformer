import torch
import torch.nn as nn
import random, numpy as np
from model import DecoderOnlyTransformer
from my_auto_tokenizer import MyAutoTokenizer

def train():
    # 固定随机种子 设定种子后，每次跑都能得到可复现的结果
    torch.manual_seed(43)
    random.seed(42)
    np.random.seed(42)

    # ========= 数据准备 =========
    text = "hello transformer! what is 5 + 9 ?"
    tokenizer = MyAutoTokenizer(level="char", texts=[text])

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # (seq_len,)
    
    # print("Original data:", original_data)
    # print("Tensor:", data)
    # print("Tensor storage dtype:", data.dtype)
    # print("Tensor shape:", data.shape)
    # print("Tensor device:", data.device)
    # print("Requires grad:", data.requires_grad)   # 是否跟踪梯度
    # print("Is leaf:", data.is_leaf)               # 是否是叶子节点（可用于优化器更新）
    # print("Number of elements:", data.numel())    # 总元素个数
    # print("Size:", data.size())                   # 等价于 shape
    # print("Dimensions:", data.dim())              # 维度个数

    # 输入 x = 前 n-1 个 token, 目标 y = 后 n-1 个 token
    x = data[:-1].unsqueeze(0)  # (1, T-1)
    y = data[1:].unsqueeze(0)   # (1, T-1)

    vocab_size = tokenizer.vocab_size

     # ========= 模型 & 优化器 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=128, max_len=128)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x, y = x.to(device), y.to(device)

    # ========= 训练循环 =========
    for step in range(5):  # 跑 200 步看看 loss 能否下降
        optimizer.zero_grad()

        logits = model(x)                # (B, T, vocab_size)
        loss = criterion(
            logits.view(-1, vocab_size), # (B*T, vocab_size)
            y.view(-1)                   # (B*T,)
        )

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step} | loss = {loss.item():.4f}")

    # ========= 测试预测 =========
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=-1)  # (B,T)
        print("输入:", tokenizer.decode(x[0].tolist()))
        print("预测:", tokenizer.decode(pred[0].tolist()))
        print("目标:", tokenizer.decode(y[0].tolist()))

if __name__ == "__main__":
    train() 