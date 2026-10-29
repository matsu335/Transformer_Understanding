# Transformer_Understanding

Transformerを数式レベルからnumpyのみで実装して理解するためのリポジトリです。

## 記事
- ① Forward Propagation編：https://zenn.dev/matsu8/articles/909ccb2144b1ae

## ファイル構成
- `code/1_functions_forward.py`：必要関数（Linear, Embedding, Multi-Head Attentionなど）
- `code/1_architecture_forward.py`：Encoder / Decoder / Transformerの組み立て
- `code/1_test_forward.py`：Forward Propagationの動作確認

## 実行方法
```
pip install -r requirements.txt
python code/1_test_forward.py
```
