use std::{fs::File, io::Read, path::Path};

use crate::tokenizer::{BPETokenizer, END_TOKEN, train_bpe};

mod tokenizer;

use ciborium::into_writer;

fn training_tokenizer(
    input_path: &Path,
    output_path: &Path,
    vocab_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut input_file = File::open(input_path)?;

    let mut text = String::new();
    input_file.read_to_string(&mut text)?;

    let merge_rules = train_bpe(&text, vocab_size);
    let output_file = File::create(output_path)?;
    Ok(into_writer(&merge_rules, output_file)?)
}

fn ch01_training_tokenizer() {
    let input_path = Path::new("./data/tiny_codes.txt");
    let output_path = Path::new("./data/merge_rules.cbor");
    let vocab_size = 1000;

    training_tokenizer(input_path, output_path, vocab_size).unwrap();
}

fn ch01_tokenizer_check() {
    let output_path = Path::new("./data/merge_rules.cbor");
    let vocab_size = 1000;

    let tokenizer = BPETokenizer::load_from(output_path, END_TOKEN).unwrap();

    for id in 256..280 {
        println!("{} -> '{}'", id, tokenizer.decode(&[id]));
    }

    for id in 990..vocab_size {
        println!("{} ->  '{}'", id, tokenizer.decode(&[id]));
    }
}

fn main() {
    ch01_training_tokenizer();
    ch01_tokenizer_check();
}
