import numpy as np


# ============================================================
# 0. Constants
# ============================================================

B = 2 # batch size. 一度に処理する数列の数
L = 6 # sequence length. 一つの数列あたりのシーケンス長
d_model = 64 # トークン表現の次元数
num_heads = 4 # Attention Headの数
d_k = d_model // num_heads # 各HeadのQuery/Key/Valueの次元数
d_ff = 256 # FFN中間層の次元数
vocab_size = 13 # Vocabulary Size. 0~9, <PAD>, <BOS>, <EOS>の13種類を想定


# ============================================================
# 1. Linear
# ============================================================

class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))

        self.weight = np.random.uniform(
            -limit, limit,
            (in_features, out_features)
        )

        self.bias = np.zeros(out_features)

    def forward(self, x):
        return x @ self.weight + self.bias


# ============================================================
# 2. Embedding
# ============================================================

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

    def forward(self, token_ids):
        return self.weight[token_ids] * np.sqrt(self.d_model)                                                                                                                                                                                                                       


# ============================================================
# 3. Positional Encoding
# ============================================================

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, None]

    div_term = np.exp(
        np.arange(0, d_model, 2)
        * (-np.log(10000.0) / d_model)
    )

    PE[:, 0::2] = np.sin(position * div_term)

    PE[:, 1::2] = np.cos(position * div_term)

    return PE


# ============================================================
# 4. Causal Mask
# ============================================================

def causal_mask(seq_len):
    return np.tril(
        np.ones((seq_len, seq_len), dtype=bool)
    )


# ============================================================
# 5. Softmax
# ============================================================

def softmax(X, axis=-1):
    X_shifted = X - np.max(
        X, axis=axis, keepdims=True
    )

    exp_X = np.exp(X_shifted)

    return exp_X / np.sum(
        exp_X, axis=axis, keepdims=True
    )


# ============================================================
# 6. Multi-Head Attention
# ============================================================

# Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, d_k):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
    def split_heads(self, X):
        B, L, D = X.shape

        X = X.reshape(B, L, self.num_heads, self.d_k)

        return X.transpose(0, 2, 1, 3)

    def combine_heads(self, X):
        B, H, L, D = X.shape

        return X.transpose(0, 2, 1, 3).reshape(B, L, H * D)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q.forward(query))
        K = self.split_heads(self.W_k.forward(key))
        V = self.split_heads(self.W_v.forward(value))

        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(output)

        return self.W_o.forward(output), weights


# ============================================================
# 7. Scaled Dot-Product Attention
# ============================================================

# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = Q @ K.swapaxes(-2, -1)
    scores = scores / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)

    output = weights @ V
    return output, weights


# ============================================================
# 8. Layer Normalization
# ============================================================

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)

        var = np.var(X, axis=-1, keepdims=True)

        X_norm = (X - mean) / np.sqrt(var + self.eps)

        return self.gamma * X_norm + self.beta 


# ============================================================
# 9. Feed Forward Network
# ============================================================

def relu(X):
    return np.maximum(0, X)
    
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, X):
        hidden = self.linear1.forward(X)
        hidden = relu(hidden)

        return self.linear2.forward(hidden)
