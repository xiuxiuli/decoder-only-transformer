import string
# ====== 简化版 Tokenizer ======
class MyAutoTokenizer:
    def __init__(self, level="char", texts=None):
        # 固定词表（字符级）
        self.vocab = list(string.ascii_lowercase + " !?.,")

        # 加两个特殊符号：未知 & 填充
        self.vocab = ["<unk>", "<pad>"] + self.vocab

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = { i: ch for ch, i in self.stoi.items()}

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def pad_token_id(self):
        return self.stoi[self.pad_token]

    @property
    def unk_token_id(self):
        return self.stoi[self.unk_token]
    
    # Tokenizer 干的事就是：文本 ↔ 索引序列的双向映射
    def encode(self, text, max_len=None, padding=False):
        # 查每个字母的索引，如果不存在就用 <unk> 的索引
        # [字母的索引, 字母的索引, ....字母的索引....]
        ids = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text.lower()]

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[: max_len]
            if padding:
                ids = ids + [self.stoi[self.pad_token]] * (max_len - len(ids))
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for i in ids:
            tok  = self.itos.get(i, self.unk_token)
            if skip_special_tokens and tok in[self.pad_token, self.unk_token]:
                continue
            tokens.append(tok)
        return "".join(tokens)
    





        