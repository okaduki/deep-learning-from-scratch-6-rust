use std::{fs::File, path::Path};

use crate::tokenizer::{BPETokenizer, END_TOKEN, train_bpe};

mod tokenizer;

use ciborium::into_writer;

fn training_tokenizer() {}

fn main() {
    let sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!";
    let vocab_size = 280;

    if true {
        let merge_rules = train_bpe(sample_text, vocab_size);
        let file = File::create("rules.cbor").unwrap();
        into_writer(&merge_rules, file).unwrap();
    } else {
        let tokenizer = BPETokenizer::load_from(&Path::new("rules.cbor"), END_TOKEN).unwrap();

        let text = "Say hello!";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        dbg!(&ids);
        dbg!(&decoded);
    }
}
