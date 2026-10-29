import importlib

import numpy as np

# ============================================================
# 0. Imports
# ============================================================

# 数字で始まるファイル名はimport文で読み込めないため、importlibを使う
functions = importlib.import_module("1_functions_forward")
architecture = importlib.import_module("1_architecture_forward")

B = functions.B
L = functions.L
vocab_size = functions.vocab_size
Transformer = architecture.Transformer


# ============================================================
# 1. Forward Test
# ============================================================

if __name__ == "__main__":
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

    logits = model.forward(src_tokens, tgt_tokens)

    print("Logits shape:", logits.shape)

    assert logits.shape == (B, L, vocab_size)
    assert np.all(np.isfinite(logits))

    print("Forward test passed.")
