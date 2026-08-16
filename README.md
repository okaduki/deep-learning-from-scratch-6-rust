# deep-learning-from-scratch-6-rust

『ゼロから作る Deep Learning 6』の CodeBot 例を Rust と [Burn](https://burn.dev/) で実装した学習用プロジェクトです。

現在は CUDA backend を使用し、次の流れを実装しています。

1. BPE トークナイザーの学習
2. コードデータによる事前学習
3. 指示応答データによる SFT
4. 足し算問題を報酬とする GRPO

## 必要環境

- Rust 1.85 以降（edition 2024）
- NVIDIA GPU と CUDA driver

Burn は Cargo が依存関係を解決するときに CUDA 関連ライブラリをビルドします。CUDA を利用できない環境では、このままでは実行できません。

## データ

| ファイル | 用途 |
| --- | --- |
| `data/tiny_codes.txt` | BPE 学習と事前学習の元データ |
| `data/merge_rules.cbor` | 学習済み BPE merge rule |
| `data/tiny_codes.bin` | 事前学習用にトークン ID 化したデータ |
| `data/tiny_codes_sft.json` | SFT 用の instruction/response データ |

`merge_rules.cbor` と `tiny_codes.bin` がない場合は、以下の「実行順序」の 1 と 2 を先に実行してください。

## 実行順序

実行する関数は [src/main.rs](src/main.rs) の `main()` で切り替えます。一度に有効にする呼び出しは一つだけにしてください。

### 1. BPE トークナイザーを学習する

```rust
fn main() {
    ch01_training_tokenizer();
}
```

```bash
cargo run --release
```

`data/merge_rules.cbor` が生成されます。

### 2. 事前学習データをトークン化する

```rust
fn main() {
    ch01_convert_binary();
}
```

```bash
cargo run --release
```

`data/tiny_codes.bin` が生成されます。

### 3. 事前学習する

```rust
fn main() {
    ch03_training();
}
```

```bash
cargo run --release
```

学習済みモデルは `artifacts/ch02/final_model.mpk` に保存されます。既定では最大 20,000 iteration です。短い確認には環境変数を使えます。

```bash
CH02_ITERS=100 CH02_EPOCHS=1 cargo run --release
```

### 4. SFT を実行する

```rust
fn main() {
    ch03_sft();
}
```

```bash
cargo run --release
```

SFT は約 1,500 update 実行し、`artifacts/ch02/model_sft.mpk` に保存します。

### 5. GRPO を実行する

```rust
fn main() {
    ch03_grpo();
}
```

```bash
cargo run --release
```

GRPO は 1 から 9 の足し算 81 問を使い、各 prompt から 8 個の応答を生成します。`artifacts/ch02/model_grpo.mpk` に保存されます。10 iteration ごとに正解率と生成サンプルを表示します。

## 生成とチャット

事前学習済みモデルで生成する場合:

```rust
fn main() {
    ch03_generate();
}
```

SFT 済みモデルで対話する場合:

```rust
fn main() {
    ch03_chat();
}
```

いずれも `cargo run --release` で実行後、標準入力から prompt を与えます。

## Checkpoint について

モデルは `SFTTrainer { policy: GPT }` の record として保存します。`final_model.mpk` と `model_sft.mpk` はこの形式を前提に読み込みます。

LayerNorm の初期化や checkpoint 読み込み形式を変更した場合は、古い artifact を混在させず、事前学習、SFT、GRPO の順に再生成してください。

## 検証

```bash
cargo fmt --check
cargo test
cargo build --release
```
