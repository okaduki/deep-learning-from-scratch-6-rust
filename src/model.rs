use core::f32;

use burn::{
    module::{Initializer, Param},
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::Tensor,
    tensor::activation::softmax,
    tensor::backend::Backend,
    tensor::module::linear,
};

use burn::module::Module;

const INIT_STD: f64 = 0.02;

fn init_embedding<B: Backend>(
    n_embedding: usize,
    d_model: usize,
    device: &B::Device,
) -> Embedding<B> {
    EmbeddingConfig::new(n_embedding, d_model)
        .with_initializer(Initializer::Normal {
            mean: 0.0,
            std: INIT_STD,
        })
        .init(device)
}

fn init_linear<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    let mut linear = LinearConfig::new(d_input, d_output)
        .with_initializer(Initializer::Normal {
            mean: 0.0,
            std: INIT_STD,
        })
        .init(device);

    linear.bias = Some(Initializer::Zeros.init([d_output], device));
    linear
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    embed: Embedding<B>,
    pos_embed: Embedding<B>,
    dropout: Dropout,

    blocks: Vec<Block<B>>,

    norm: LayerNorm<B>,
    unembed_bias: Param<Tensor<B, 1>>,
}

#[derive(Config, Debug)]
pub struct GPTConfig {
    vocab_size: usize,
    max_context_len: usize,
    embed_dim: usize,
    n_head: usize,
    n_layer: usize,
    ff_dim: usize,
    dropout_rate: f64,
}

impl GPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT<B> {
        let mut blocks = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            blocks.push(
                BlockConfig::new(self.embed_dim, self.n_head)
                    .with_ff_dim(Some(self.ff_dim))
                    .with_dropout_rate(self.dropout_rate)
                    .init(device),
            );
        }

        let gpt = GPT {
            embed: init_embedding(self.vocab_size, self.embed_dim, device),
            pos_embed: init_embedding(self.max_context_len, self.embed_dim, device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            blocks: blocks,
            norm: LayerNormConfig::new(self.embed_dim).init(device),
            unembed_bias: Initializer::Zeros.init([self.vocab_size], device),
        };

        gpt
    }
}

impl<B: Backend> GPT<B> {
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3, Float> {
        let [_batch_size, context_len] = ids.dims();
        let device = ids.device();

        let pos = Tensor::arange(0..context_len as i64, &device);
        let pos_emb = self.pos_embed.forward(pos.reshape([1, context_len]));
        let emb = self.embed.forward(ids);

        let mut x = self.dropout.forward(pos_emb + emb);

        for block in self.blocks.iter() {
            x = block.forward(x);
        }
        x = self.norm.forward(x);

        let logits = linear(
            x,
            self.embed.weight.val().transpose(),
            Some(self.unembed_bias.val()),
        );
        logits
    }
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ffn: FFN<B>,
}

#[derive(Config, Debug)]
struct BlockConfig {
    embed_dim: usize,
    n_head: usize,
    ff_dim: Option<usize>,

    #[config(default = "0.1")]
    dropout_rate: f64,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        let head_dim = self.embed_dim / self.n_head;

        Block {
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            attention: MultiHeadAttentionConfig::new(self.n_head, head_dim, self.embed_dim)
                .init(device),
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            ffn: FFNConfig::new(self.embed_dim)
                .with_hidden_dim(self.ff_dim)
                .with_dropout_rate(self.dropout_rate)
                .init(device),
        }
    }
}

impl<B: Backend> Block<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attention.forward(self.norm1.forward(x));
        let x = x.clone() + self.ffn.forward(self.norm2.forward(x));
        x
    }
}

#[derive(Module, Debug)]
struct MultiHeadAttention<B: Backend> {
    w_q: Linear<B>,
    w_k: Linear<B>,
    w_v: Linear<B>,
    w_o: Linear<B>,

    attention_dropout: Dropout,
    output_dropout: Dropout,

    n_head: usize,
    head_dim: usize,
    embed_dim: usize,
}

#[derive(Config, Debug)]
struct MultiHeadAttentionConfig {
    n_head: usize,
    head_dim: usize,
    embed_dim: usize,

    #[config(default = "0.1")]
    dropout_rate: f64,
}

impl MultiHeadAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        MultiHeadAttention {
            w_q: init_linear(self.embed_dim, self.n_head * self.head_dim, device),
            w_k: init_linear(self.embed_dim, self.n_head * self.head_dim, device),
            w_v: init_linear(self.embed_dim, self.n_head * self.head_dim, device),
            w_o: init_linear(self.n_head * self.head_dim, self.embed_dim, device),
            attention_dropout: DropoutConfig::new(self.dropout_rate).init(),
            output_dropout: DropoutConfig::new(self.dropout_rate).init(),

            n_head: self.n_head,
            head_dim: self.head_dim,
            embed_dim: self.embed_dim,
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, context_size, _embed_size] = x.dims(); // [B,C,E]

        let q = self.w_q.forward(x.clone()); // [B,C,H*D]
        let k = self.w_k.forward(x.clone()); // [B,C,H*D]
        let v = self.w_v.forward(x); // [B,C,H*D]

        let q = q
            .reshape([batch_size, context_size, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B,H,C,D]
        let k = k
            .reshape([batch_size, context_size, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B,H,C,D]
        let v = v
            .reshape([batch_size, context_size, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B,H,C,D]

        let kt = k.swap_dims(2, 3); // [B,H,D,C]
        let scores = q.matmul(kt); // [B,H,C,C]
        let scores = scores / (self.head_dim as f32).sqrt();

        let mask: Tensor<B, 4> =
            Tensor::<B, 2>::ones(Shape::new([context_size, context_size]), &scores.device())
                .tril(0)
                .reshape([1, 1, context_size, context_size]);
        let scores = scores.mask_fill(mask.equal_elem(1), f32::NEG_INFINITY);

        let weights = softmax(scores, 3); // [B,H,C,C]
        let weights = self.attention_dropout.forward(weights);
        let hidden = weights.matmul(v); // [B,H,C,D]

        let hidden = hidden.swap_dims(1, 2); // [B,C,H,D]
        let hidden = hidden.reshape([batch_size, context_size, self.n_head * self.head_dim]); // [B,C,H*D]

        let output = self.w_o.forward(hidden); // [B,C,D]
        let output = self.output_dropout.forward(output);

        output
    }
}

#[derive(Module, Debug)]
struct LayerNorm<B: Backend> {
    embed_dim: usize,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    eps: f32,
}

#[derive(Config, Debug)]
struct LayerNormConfig {
    embed_dim: usize,

    #[config(default = "1e-5")]
    eps: f32,
}

impl LayerNormConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm {
            embed_dim: self.embed_dim,
            gamma: Param::from_tensor(Tensor::ones(Shape::new([self.embed_dim]), device)),
            beta: Param::from_tensor(Tensor::ones(Shape::new([self.embed_dim]), device)),
            eps: self.eps,
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (var, mean) = x.clone().var_mean(2);
        let norm_x = (x - mean) / (var + self.eps).sqrt();

        let gamma = self.gamma.val().reshape([1, 1, self.embed_dim]);
        let beta = self.beta.val().reshape([1, 1, self.embed_dim]);
        gamma * norm_x + beta
    }
}

#[derive(Module, Debug, Clone)]
struct GELU {}

impl GELU {
    fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y: Tensor<B, 3> =
            (2. / f32::consts::PI).sqrt() * (x.clone() + 0.044715 * x.clone().powf_scalar(3.));
        0.5 * x * (1. + y.tanh())
    }
}

#[derive(Module, Debug)]
struct FFN<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    gelu: GELU,
    dropout: Dropout,
}

#[derive(Config, Debug)]
struct FFNConfig {
    x_dim: usize,
    hidden_dim: Option<usize>,

    #[config(default = "0.1")]
    dropout_rate: f64,
}

impl FFNConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FFN<B> {
        let hidden_dim = self.hidden_dim.unwrap_or(4 * self.x_dim);

        FFN {
            linear1: init_linear(self.x_dim, hidden_dim, device),
            linear2: init_linear(hidden_dim, self.x_dim, device),
            gelu: GELU {},
            dropout: DropoutConfig::new(self.dropout_rate).init(),
        }
    }
}

impl<B: Backend> FFN<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear2.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}
