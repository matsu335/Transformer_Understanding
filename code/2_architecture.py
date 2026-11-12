import importlib

# ============================================================
# 0. Imports
# ============================================================

# 数字で始まるファイル名はimport文で読み込めないため、importlibを使う
functions = importlib.import_module("2_functions")

L = functions.L
d_model = functions.d_model
num_heads = functions.num_heads
d_k = functions.d_k
d_ff = functions.d_ff
vocab_size = functions.vocab_size
Linear = functions.Linear
Embedding = functions.Embedding
MultiHeadAttention = functions.MultiHeadAttention
FeedForward = functions.FeedForward
LayerNorm = functions.LayerNorm
positional_encoding = functions.positional_encoding
causal_mask = functions.causal_mask


# ============================================================
# 1. Encoder Layer
# ============================================================

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_k, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads, d_k)
        self.norm1 = LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X, src_mask=None):
        # Self-Attention
        attention_output, _ = self.self_attention.forward(
            X, X, X, mask=src_mask
        )

        # Residual Connection + LayerNorm
        X = self.norm1.forward(X + attention_output)

        # Feed Forward Network
        ff_output = self.feed_forward.forward(X)

        # Residual Connection + LayerNorm
        X = self.norm2.forward(X + ff_output)

        return X

    def backward(self, dX):
        # norm2(X_ff + ff_output) の逆伝播
        d_residual2 = self.norm2.backward(dX)
        d_ff_input = d_residual2  # residual 経由
        d_ff_output = d_residual2

        d_ff_input = d_ff_input + self.feed_forward.backward(d_ff_output)

        # norm1(X_attn + attention_output) の逆伝播
        d_residual1 = self.norm1.backward(d_ff_input)
        d_attn_input = d_residual1  # residual 経由
        d_attn_output = d_residual1

        dQ, dK, dV = self.self_attention.backward(d_attn_output)
        # Self-Attention: Q=K=V=X なので勾配を合算
        d_attn_input = d_attn_input + dQ + dK + dV

        return d_attn_input

    def zero_grad(self):
        self.self_attention.zero_grad()
        self.norm1.zero_grad()
        self.feed_forward.zero_grad()
        self.norm2.zero_grad()


# ============================================================
# 2. Encoder
# ============================================================

class Encoder:
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_k,
        d_ff,
        num_layers,
        max_len,
    ):
        self.embedding = Embedding(vocab_size, d_model)

        self.pos_encoding = positional_encoding(max_len, d_model)

        self.layers = [
            EncoderLayer(d_model, num_heads, d_k, d_ff)
            for _ in range(num_layers)
        ]

    def forward(self, src_tokens, src_mask=None):
        # Token Embedding
        X = self.embedding.forward(src_tokens)

        # Positional Encoding
        seq_len = src_tokens.shape[1]
        X = X + self.pos_encoding[None, :seq_len, :]

        # Encoder Layers
        for layer in self.layers:
            X = layer.forward(X, src_mask)

        return X

    def backward(self, dX):
        # レイヤを逆順に逆伝播
        for layer in reversed(self.layers):
            dX = layer.backward(dX)

        # PE は定数なので勾配はそのまま Embedding へ
        self.embedding.backward(dX)
        return None

    def zero_grad(self):
        self.embedding.zero_grad()
        for layer in self.layers:
            layer.zero_grad()


# ============================================================
# 3. Decoder Layer
# ============================================================

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_k, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads, d_k)
        self.norm1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, d_k)
        self.norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def forward(
        self,
        X,
        encoder_output,
        tgt_mask=None,
        src_mask=None,
    ):
        # Masked Self-Attention
        self_attn_output, _ = self.self_attention.forward(
            X, X, X, mask=tgt_mask
        )

        X = self.norm1.forward(X + self_attn_output)

        # Encoder-Decoder Cross-Attention
        cross_attn_output, _ = self.cross_attention.forward(
            query=X,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        X = self.norm2.forward(X + cross_attn_output)

        # Feed Forward Network
        ff_output = self.feed_forward.forward(X)

        X = self.norm3.forward(X + ff_output)

        return X

    def backward(self, dX):
        # --- FFN + residual + norm3 ---
        d_residual3 = self.norm3.backward(dX)
        d_ff_input = d_residual3
        d_ff_output = d_residual3
        d_ff_input = d_ff_input + self.feed_forward.backward(d_ff_output)

        # --- Cross-Attention + residual + norm2 ---
        d_residual2 = self.norm2.backward(d_ff_input)
        d_cross_query = d_residual2
        d_cross_output = d_residual2

        dQ, dK, dV = self.cross_attention.backward(d_cross_output)
        d_cross_query = d_cross_query + dQ
        # encoder_output への勾配 (K と V)
        d_encoder = dK + dV

        # --- Self-Attention + residual + norm1 ---
        d_residual1 = self.norm1.backward(d_cross_query)
        d_self_input = d_residual1
        d_self_output = d_residual1

        dQ, dK, dV = self.self_attention.backward(d_self_output)
        d_self_input = d_self_input + dQ + dK + dV

        return d_self_input, d_encoder

    def zero_grad(self):
        self.self_attention.zero_grad()
        self.norm1.zero_grad()
        self.cross_attention.zero_grad()
        self.norm2.zero_grad()
        self.feed_forward.zero_grad()
        self.norm3.zero_grad()


# ============================================================
# 4. Decoder
# ============================================================

class Decoder:
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_k,
        d_ff,
        num_layers,
        max_len,
    ):
        self.embedding = Embedding(vocab_size, d_model)

        self.pos_encoding = positional_encoding(max_len, d_model)

        self.layers = [
            DecoderLayer(d_model, num_heads, d_k, d_ff)
            for _ in range(num_layers)
        ]

    def forward(
        self,
        tgt_tokens,
        encoder_output,
        tgt_mask=None,
        src_mask=None,
    ):
        # Token Embedding
        X = self.embedding.forward(tgt_tokens)

        # Positional Encoding
        seq_len = tgt_tokens.shape[1]
        X = X + self.pos_encoding[None, :seq_len, :]

        # Decoder Layers
        for layer in self.layers:
            X = layer.forward(
                X,
                encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
            )

        return X

    def backward(self, dX):
        d_encoder_total = 0.0

        for layer in reversed(self.layers):
            dX, d_encoder = layer.backward(dX)
            d_encoder_total = d_encoder_total + d_encoder

        # PE は定数なので勾配はそのまま Embedding へ
        self.embedding.backward(dX)

        return d_encoder_total

    def zero_grad(self):
        self.embedding.zero_grad()
        for layer in self.layers:
            layer.zero_grad()


# ============================================================
# 5. Transformer
# ============================================================

class Transformer:
    def __init__(
        self,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_k=d_k,
        d_ff=d_ff,
        num_layers=2,
        max_len=L,
        pad_id=0,
    ):
        self.pad_id = pad_id

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_heads,
            d_k,
            d_ff,
            num_layers,
            max_len,
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            d_k,
            d_ff,
            num_layers,
            max_len,
        )

        self.output_linear = Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src_tokens):
        # (B, 1, 1, L_src)
        return (src_tokens != self.pad_id)[:, None, None, :]

    def make_tgt_mask(self, tgt_tokens):
        B, L_tgt = tgt_tokens.shape

        # Padding Mask: (B, 1, 1, L_tgt)
        padding_mask = (
            tgt_tokens != self.pad_id
        )[:, None, None, :]

        # Causal Mask: (1, 1, L_tgt, L_tgt)
        future_mask = causal_mask(L_tgt)[None, None, :, :]

        # (B, 1, L_tgt, L_tgt)
        return padding_mask & future_mask

    def forward(self, src_tokens, tgt_tokens):
        src_mask = self.make_src_mask(src_tokens)
        tgt_mask = self.make_tgt_mask(tgt_tokens)

        # Encoder
        encoder_output = self.encoder.forward(
            src_tokens,
            src_mask=src_mask,
        )

        # Decoder
        decoder_output = self.decoder.forward(
            tgt_tokens,
            encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
        )

        # Vocabulary Projection
        logits = self.output_linear.forward(decoder_output)

        return logits

    def backward(self, d_logits):
        # output_linear
        d_decoder = self.output_linear.backward(d_logits)

        # decoder -> encoder_output への勾配も返る
        d_encoder = self.decoder.backward(d_decoder)

        # encoder
        self.encoder.backward(d_encoder)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.output_linear.zero_grad()
