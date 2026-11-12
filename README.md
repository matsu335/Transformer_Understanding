# Transformer_Understanding

Transformerを数式レベルからnumpyのみで実装して理解するためのリポジトリです。

## 記事
- ① Forward Propagation編：https://zenn.dev/matsu8/articles/909ccb2144b1ae
- ② Back Propagation編：https://zenn.dev/matsu8/articles/a01ba3fc713e03

## ファイル構成
### ① Forward Propagation編
- `code/1_functions_forward.py`：必要関数（Linear, Embedding, Multi-Head Attentionなど）
- `code/1_architecture_forward.py`：Encoder / Decoder / Transformerの組み立て
- `code/1_test_forward.py`：Forward Propagationの動作確認

### ② Back Propagation編
- `code/2_functions.py`：必要関数（①の関数にbackwardを追加したもの＋交差エントロピー損失）
- `code/2_architecture.py`：Encoder / Decoder / Transformerの組み立て（backwardを追加したもの）
- `code/2_test.py`：Forward / Back Propagationの動作確認

## 実行方法
```
pip install -r requirements.txt

# ① Forward Propagation編
python code/1_test_forward.py

# ② Back Propagation編
python code/2_test.py
```
