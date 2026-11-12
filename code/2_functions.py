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

        self.dW = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

    def forward(self, x):
        self.cache = x
        return x @ self.weight + self.bias

    def backward(self, dY):
        x = self.cache

        # (..., in) @ (in, out) の勾配
        x_2d = x.reshape(-1, x.shape[-1])
        dY_2d = dY.reshape(-1, dY.shape[-1])

        self.dW = x_2d.T @ dY_2d
        self.db = np.sum(dY_2d, axis=0)

        return dY @ self.weight.T

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)


# ============================================================
# 2. Embedding
# ============================================================

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.dW = np.zeros_like(self.weight)

    def forward(self, token_ids):
        self.cache = token_ids
        return self.weight[token_ids] * np.sqrt(self.d_model)

    def backward(self, dY):
        token_ids = self.cache
        scale = np.sqrt(self.d_model)

        self.dW.fill(0.0)
        np.add.at(
            self.dW,
            token_ids.ravel(),
            (dY * scale).reshape(-1, self.d_model),
        )

        # Embedding の入力は離散トークンなので勾配は返さない
        return None

    def zero_grad(self):
        self.dW.fill(0.0)


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


def softmax_backward(dY, Y, axis=-1):
    # Y = softmax(X) のとき
    # dX_i = Y_i * (dY_i - sum_j(dY_j * Y_j))
    sum_term = np.sum(dY * Y, axis=axis, keepdims=True)
    return Y * (dY - sum_term)


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

        output, weights, attn_cache = scaled_dot_product_attention(
            Q, K, V, mask
        )

        combined = self.combine_heads(output)

        self.cache = {
            "attn_cache": attn_cache,
            "head_shape": output.shape,  # (B, H, Lq, Dk)
        }

        return self.W_o.forward(combined), weights

    def backward(self, d_output):
        # W_o
        d_combined = self.W_o.backward(d_output)

        # combine_heads の逆: (B, L, H*D) -> (B, H, L, D)
        B, H, Lq, Dk = self.cache["head_shape"]
        d_heads = d_combined.reshape(B, Lq, H, Dk).transpose(0, 2, 1, 3)

        dQ, dK, dV = scaled_dot_product_attention_backward(
            d_heads, self.cache["attn_cache"]
        )

        # split_heads の逆: (B, H, L, D) -> (B, L, H*D)
        # Cross-Attention では Q と K/V で系列長が異なる
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, Lq, H * Dk)

        Lk = self.cache["attn_cache"]["K"].shape[2]
        Lv = self.cache["attn_cache"]["V"].shape[2]
        dK = dK.transpose(0, 2, 1, 3).reshape(B, Lk, H * Dk)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, Lv, H * Dk)

        d_query = self.W_q.backward(dQ)
        d_key = self.W_k.backward(dK)
        d_value = self.W_v.backward(dV)

        return d_query, d_key, d_value

    def zero_grad(self):
        self.W_q.zero_grad()
        self.W_k.zero_grad()
        self.W_v.zero_grad()
        self.W_o.zero_grad()


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

    # 全マスク行などで生じる NaN を 0 に置換
    weights = np.nan_to_num(weights, nan=0.0)

    output = weights @ V

    cache = {
        "Q": Q,
        "K": K,
        "V": V,
        "weights": weights,
        "mask": mask,
        "d_k": d_k,
    }

    return output, weights, cache


def scaled_dot_product_attention_backward(d_output, cache):
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    weights = cache["weights"]
    mask = cache["mask"]
    d_k = cache["d_k"]

    # output = weights @ V
    d_weights = d_output @ V.swapaxes(-2, -1)
    dV = weights.swapaxes(-2, -1) @ d_output

    # weights = softmax(scores)
    d_scores = softmax_backward(d_weights, weights, axis=-1)

    if mask is not None:
        d_scores = np.where(mask, d_scores, 0.0)

    # scores = (Q @ K^T) / sqrt(d_k)
    scale = 1.0 / np.sqrt(d_k)
    dQ = (d_scores @ K) * scale
    dK = (d_scores.swapaxes(-2, -1) @ Q) * scale

    return dQ, dK, dV


# ============================================================
# 8. Layer Normalization
# ============================================================

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)

        var = np.var(X, axis=-1, keepdims=True)

        std = np.sqrt(var + self.eps)
        X_norm = (X - mean) / std

        self.cache = {
            "X_norm": X_norm,
            "std": std,
        }

        return self.gamma * X_norm + self.beta

    def backward(self, dY):
        X_norm = self.cache["X_norm"]
        std = self.cache["std"]

        # パラメータ勾配
        reduce_axes = tuple(range(dY.ndim - 1))
        self.dgamma = np.sum(dY * X_norm, axis=reduce_axes)
        self.dbeta = np.sum(dY, axis=reduce_axes)

        # 入力勾配
        # dX = (1/std) * (dX_norm - mean(dX_norm) - X_norm * mean(dX_norm * X_norm))
        dX_norm = dY * self.gamma

        dX = (
            (1.0 / std)
            * (
                dX_norm
                - np.mean(dX_norm, axis=-1, keepdims=True)
                - X_norm * np.mean(
                    dX_norm * X_norm, axis=-1, keepdims=True
                )
            )
        )

        return dX

    def zero_grad(self):
        self.dgamma.fill(0.0)
        self.dbeta.fill(0.0)


# ============================================================
# 9. Feed Forward Network
# ============================================================

def relu(X):
    return np.maximum(0, X)


def relu_backward(dY, X):
    return dY * (X > 0)


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, X):
        hidden = self.linear1.forward(X)
        self.cache_pre_relu = hidden
        hidden = relu(hidden)

        return self.linear2.forward(hidden)

    def backward(self, dY):
        d_hidden = self.linear2.backward(dY)
        d_hidden = relu_backward(d_hidden, self.cache_pre_relu)
        return self.linear1.backward(d_hidden)

    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()


# ============================================================
# 10. Cross Entropy Loss (学習・勾配確認用)
# ============================================================

def cross_entropy_loss(logits, targets, pad_id=0):
    """
    logits:  (B, L, V)
    targets: (B, L)
    PAD 位置は損失・勾配から除外する。
    """
    B, L, V = logits.shape

    # 数値安定な softmax
    logits_shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shift)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    flat_probs = probs.reshape(B * L, V)
    flat_targets = targets.reshape(B * L)
    selected = flat_probs[np.arange(B * L), flat_targets].reshape(
        B, L
    )

    mask = (targets != pad_id).astype(np.float64)
    n_valid = np.maximum(mask.sum(), 1.0)

    loss = -np.sum(np.log(selected + 1e-12) * mask) / n_valid

    # dL/dlogits
    d_logits = probs.copy()
    d_logits.reshape(B * L, V)[
        np.arange(B * L), flat_targets
    ] -= 1.0
    d_logits = d_logits * mask[..., None] / n_valid

    return loss, d_logits
