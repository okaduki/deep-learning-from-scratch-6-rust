#![recursion_limit = "256"]
#![allow(unused_imports)]
use std::fs;
use std::{fs::File, io::Read, path::Path};

use crate::tokenizer::{BPETokenizer, END_TOKEN, train_bpe};
use crate::training::{chat, generate};

#[allow(dead_code)]
mod model;
#[allow(dead_code)]
mod tokenizer;
#[allow(dead_code)]
mod training;

use ciborium::{from_reader, into_writer};

#[allow(dead_code)]
fn training_tokenizer(
    input_path: &Path,
    output_path: &Path,
    vocab_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let merge_rules = train_bpe(input_path, vocab_size);
    let output_file = File::create(output_path)?;
    Ok(into_writer(&merge_rules, output_file)?)
}

#[allow(dead_code)]
fn ch01_training_tokenizer() {
    let input_path = Path::new("./data/tiny_codes.txt");
    let output_path = Path::new("./data/merge_rules.cbor");
    let vocab_size = 1000;

    training_tokenizer(input_path, output_path, vocab_size).unwrap();
}

#[allow(dead_code)]
fn ch01_tokenizer_check() {
    let input_path = Path::new("./data/tiny_codes.txt");
    let output_path = Path::new("./data/merge_rules.cbor");
    let vocab_size = 1000;

    let tokenizer = BPETokenizer::load_from(output_path, END_TOKEN).unwrap();

    for id in 256..266 {
        println!("{} -> '{}'", id, tokenizer.decode(&[id]));
    }

    for id in 990..vocab_size {
        println!("{} ->  '{}'", id, tokenizer.decode(&[id]));
    }

    {
        let mut input_file = File::open(input_path).unwrap();
        let mut text = String::new();
        input_file.read_to_string(&mut text).unwrap();

        let original_text: String = text.chars().take(10000).collect();
        let original_binary = original_text.as_bytes();
        let encoded = tokenizer.encode(&original_text);

        println!(
            "binary size: {} / {} ({:.2} % compressed.)",
            encoded.len(),
            original_binary.len(),
            (encoded.len() as f64) / (original_binary.len() as f64) * 100.0,
        );
    }
}

#[allow(dead_code)]
fn ch01_convert_binary() {
    let input_path = Path::new("./data/tiny_codes.txt");
    let output_path = Path::new("./data/tiny_codes.bin");
    let rule_path = Path::new("./data/merge_rules.cbor");

    let tokenizer = BPETokenizer::load_from(rule_path, END_TOKEN).unwrap();

    let mut input_file = File::open(input_path).unwrap();
    let mut text = String::new();
    input_file.read_to_string(&mut text).unwrap();
    let text: String = text.chars().collect();

    let binary = tokenizer.encode(&text);
    let output_file = File::create(output_path).unwrap();
    into_writer(&binary, output_file).unwrap();
}

use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::prelude::*;
use burn::tensor::{Distribution, Int, Tensor};

#[allow(dead_code)]
fn ch02_model_check() {
    type Backend = Cuda;

    let device: CudaDevice = Default::default();

    let vocab_size = 1000;
    let max_context_len = 256;
    let embed_dim = 384;
    let n_head = 6;
    let n_layer = 6;
    let ff_dim = 4 * embed_dim;
    let dropout_rate = 0.1;
    let model = model::GPTConfig::new(
        vocab_size,
        max_context_len,
        embed_dim,
        n_head,
        n_layer,
        ff_dim,
        dropout_rate,
    )
    .init(&device);

    let distribution = Distribution::Uniform(0.0, vocab_size as f64);
    let x =
        Tensor::<Backend, 2, Int>::random(Shape::new([1, max_context_len]), distribution, &device);
    let logits = model.forward(x);
    let _ = logits;
}

#[allow(dead_code)]
fn ch03_training() {
    training::train();
}

#[allow(dead_code)]
fn ch03_generate() {
    let rule_path = Path::new("./data/merge_rules.cbor");

    let tokenizer = BPETokenizer::load_from(rule_path, END_TOKEN).unwrap();
    let mut prompt = String::new();
    println!("input:");
    std::io::stdin()
        .read_line(&mut prompt)
        .expect("failed to read line");

    let max_new_tokens = 1000;
    let temperature = 1.0;
    let resp = generate(
        &tokenizer,
        Path::new("./artifacts/ch02/final_model"),
        prompt.trim_end(),
        max_new_tokens,
        temperature,
    );

    println!("got:\n{}", resp);
}

#[allow(dead_code)]
fn ch03_chat() {
    let rule_path = Path::new("./data/merge_rules.cbor");

    let tokenizer = BPETokenizer::load_from(rule_path, END_TOKEN).unwrap();
    let mut prompt = String::new();
    println!("input:");
    std::io::stdin()
        .read_line(&mut prompt)
        .expect("failed to read line");

    let max_new_tokens = 1000;
    let temperature = 1.0;
    let resp = chat(
        &tokenizer,
        Path::new("./artifacts/ch02/model_sft"),
        prompt.trim_end(),
        max_new_tokens,
        temperature,
    );

    println!("got:\n{}", resp);
}

#[allow(dead_code)]
fn ch03_sft() {
    training::sft();
}

#[allow(dead_code)]
fn ch03_grpo() {
    training::GRPO();
}

#[allow(dead_code)]
fn ch04_training_tokenizer() {
    let input_path = Path::new("./data/tiny_stories_train.txt");
    let output_path = Path::new("./data/tiny_stories_merge_rules.cbor");
    let vocab_size = 10000;

    training_tokenizer(input_path, output_path, vocab_size).unwrap();
}

fn main() {
    // ch01_training_tokenizer();
    // ch01_tokenizer_check();
    // ch01_convert_binary();
    // ch03_training();
    // ch03_generate();
    // ch03_sft();
    // ch03_chat();
    // ch03_grpo();
    ch04_training_tokenizer();
}
