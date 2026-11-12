import importlib

import numpy as np

# ============================================================
# 0. Imports
# ============================================================

# 数字で始まるファイル名はimport文で読み込めないため、importlibを使う
functions = importlib.import_module("2_functions")
architecture = importlib.import_module("2_architecture")

B = functions.B
L = functions.L
vocab_size = functions.vocab_size
cross_entropy_loss = functions.cross_entropy_loss
Transformer = architecture.Transformer


# ============================================================
# 1. Forward / Backward Test
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)
    model = Transformer()

    # 0: PAD, 1: BOS, 2: EOS, 3-12: digits 0-9
    src_tokens = np.array([
        [6, 10, 4, 12, 5, 2],
        [7, 3, 9,  2, 0, 0],
    ])

    # Decoder inputs: BOS + shifted target
    tgt_tokens = np.array([
        [1, 5, 12, 4, 10, 6],
        [1, 9,  3, 7,  0, 0],
    ])

    # 教師信号 (次トークン予測用に 1 つずらした例)
    tgt_labels = np.array([
        [5, 12, 4, 10, 6, 2],
        [9,  3, 7,  2, 0, 0],
    ])

    # --- Forward ---
    logits = model.forward(src_tokens, tgt_tokens)

    print("Logits shape:", logits.shape)

    assert logits.shape == (B, L, vocab_size)
    assert np.all(np.isfinite(logits))

    print("Forward test passed.")

    # --- Backward ---
    loss, d_logits = cross_entropy_loss(
        logits, tgt_labels, pad_id=model.pad_id
    )

    model.zero_grad()
    model.backward(d_logits)

    print(f"Loss: {loss:.4f}")

    # 代表パラメータに勾配が入っていることを確認
    assert np.any(model.output_linear.dW != 0)
    assert np.any(model.encoder.embedding.dW != 0)
    assert np.any(model.decoder.embedding.dW != 0)
    assert np.all(np.isfinite(model.output_linear.dW))
    assert np.all(np.isfinite(model.encoder.layers[0].self_attention.W_q.dW))

    print("Backward test passed.")
