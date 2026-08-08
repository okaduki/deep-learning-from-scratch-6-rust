use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    optim::AdamWConfig,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        Device, Distribution, Int, Tensor,
        activation::softmax,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        Learner, SupervisedTraining, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use ciborium::{from_reader, into_writer};
use fancy_regex::RegexSetOptions;

use crate::model;
use std::{fmt::format, fs::File, path::Path};

use crate::tokenizer::TokenID;

#[derive(Debug, Clone)]
pub struct CodeBotItem {
    tokens: Vec<TokenID>,
    labels: Vec<TokenID>,
}

#[derive(Debug, Clone)]
pub struct CodeBotBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 2, Int>,
}

#[derive(Clone, Default)]
pub struct CodeBotBatcher;

impl<B: Backend> Batcher<B, CodeBotItem, CodeBotBatch<B>> for CodeBotBatcher {
    fn batch(&self, items: Vec<CodeBotItem>, device: &Device<B>) -> CodeBotBatch<B> {
        assert!(!items.is_empty(), "cannot create an empty batch");

        let context_size = items[0].tokens.len();
        assert!(items.iter().all(|item| {
            item.tokens.len() == context_size && item.labels.len() == context_size
        }));

        let tokens: Vec<i32> = items
            .iter()
            .flat_map(|item| item.tokens.iter().map(|&token| token as i32))
            .collect();
        let labels: Vec<i32> = items
            .iter()
            .flat_map(|item| item.labels.iter().map(|&label| label as i32))
            .collect();

        CodeBotBatch {
            tokens: Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device)
                .reshape([items.len(), context_size]),
            labels: Tensor::<B, 1, Int>::from_ints(labels.as_slice(), device)
                .reshape([items.len(), context_size]),
        }
    }
}

#[derive(Debug, Clone)]
struct CodeBotDataset {
    context_size: usize,
    tokens: Vec<TokenID>,
}

impl CodeBotDataset {
    fn new(context_size: usize, data_path: &Path) -> Self {
        let data_file = File::open(data_path).expect("dataset not found");
        let tokens: Vec<TokenID> = from_reader(data_file).expect("failed read data_file");

        CodeBotDataset {
            context_size,
            tokens,
        }
    }
}

impl Dataset<CodeBotItem> for CodeBotDataset {
    fn get(&self, index: usize) -> Option<CodeBotItem> {
        if index >= self.len() {
            return None;
        }

        let tokens = self.tokens[index..index + self.context_size].to_vec();
        let labels = self.tokens[index + 1..index + 1 + self.context_size].to_vec();

        Some(CodeBotItem { tokens, labels })
    }

    fn len(&self) -> usize {
        self.tokens.len() - self.context_size - 1
    }
}

pub fn train() {
    type TrainBackend = Autodiff<Cuda>;

    let device: CudaDevice = Default::default();
    let context_len = 256;
    let vocab_size = 1000;
    let batch_size: usize = 32;
    let learning_rate = 3e-4;
    let iterations = std::env::var("CH02_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_000);
    let num_epochs = std::env::var("CH02_EPOCHS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let embed_dim = 384;
    let n_head = 6;
    let n_layer = 6;
    let ff_dim = 4 * embed_dim;
    let dropout_rate = 0.1;

    let config = model::GPTConfig::new(
        vocab_size,
        context_len,
        embed_dim,
        n_head,
        n_layer,
        ff_dim,
        dropout_rate,
    );

    config
        .save("./artifacts/ch02/config.json")
        .expect("config save fail");

    let model = config.init::<TrainBackend>(&device);

    let data_path = Path::new("./data/tiny_codes.bin");
    let dataset = CodeBotDataset::new(context_len, data_path);
    let max_items = match std::env::var("CH02_MAX_ITEMS") {
        Ok(value) => value.parse().unwrap_or(dataset.len()),
        Err(_) => batch_size.saturating_mul(iterations).min(dataset.len()),
    }
    .min(dataset.len());
    let dataloader_train =
        DataLoaderBuilder::<TrainBackend, CodeBotItem, CodeBotBatch<TrainBackend>>::new(
            CodeBotBatcher,
        )
        .batch_size(batch_size)
        .shuffle(123)
        .set_device(device.clone())
        .build(dataset.clone())
        .slice(0, max_items);

    let dataloader_valid =
        DataLoaderBuilder::<Cuda, CodeBotItem, CodeBotBatch<Cuda>>::new(CodeBotBatcher)
            .batch_size(batch_size)
            .set_device(device.clone())
            .build(dataset)
            .slice(0, max_items);

    let optimizer = AdamWConfig::new().init::<TrainBackend, model::GPT<TrainBackend>>();
    let learner = Learner::new(model, optimizer, learning_rate);

    let result = SupervisedTraining::new("./artifacts/ch02", dataloader_train, dataloader_valid)
        .num_epochs(num_epochs)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .launch(learner);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    result
        .model
        .save_file("./artifacts/ch02/final_model", &recorder)
        .expect("failed to save the trained model");
}

pub fn generate(
    tokenizer: &crate::tokenizer::BPETokenizer,
    model_path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
) -> String {
    type TrainBackend = Autodiff<Cuda>;

    let device: CudaDevice = Default::default();

    let config =
        model::GPTConfig::load("./artifacts/ch02/config.json").expect("config file not found");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("training model not found");

    let model = config.init::<TrainBackend>(&device).load_record(record);
    let mut tokens = tokenizer.encode(prompt);

    for _ in 0..max_new_tokens {
        let x = {
            if tokens.len() <= config.max_context_len {
                tokens.clone()
            } else {
                tokens[tokens.len() - config.max_context_len..].to_vec()
            }
        };

        let item = CodeBotItem {
            tokens: x.clone(),
            labels: x,
        };

        let batcher = CodeBotBatcher::default();
        let batch = batcher.batch(vec![item], &device);
        let logits = model.forward(batch.tokens);

        let logits = logits.slice(s![.., -1, ..]).reshape([config.vocab_size]);

        let next_id: TokenID = {
            if temperature == 0.0 {
                logits.argmax(0).into_scalar() as TokenID
            } else {
                let probs = softmax(logits / temperature, 0);
                probs.categorical(1).into_scalar() as TokenID
            }
        };

        if next_id == tokenizer.end_token_id {
            break;
        }

        tokens.push(next_id);
    }

    tokenizer.decode(&tokens)
}

use crate::tokenizer::*;
use serde::{Deserialize, Serialize};
use std::fs;

const IGNORE_TOKEN: TokenID = 65535;

#[derive(Serialize, Deserialize, Debug)]
struct CodeBotSFTItem {
    instruction: String,
    response: String,
}

#[derive(Debug, Clone)]
struct CodeBotSFTDataset {
    tokens: Vec<Vec<TokenID>>,
    labels: Vec<Vec<TokenID>>,
}

fn format_prompt(instruction: &str) -> String {
    format!("### Instruction:\n{}\n\n### Response:\n", instruction)
}

impl CodeBotSFTDataset {
    fn new(context_size: usize, sft_text: &str, tokenizer: &BPETokenizer) -> Self {
        let chats: Vec<CodeBotSFTItem> = serde_json::from_str(&sft_text).expect("invalid json");

        let mut tokens = Vec::new();
        let mut labels = Vec::new();
        chats.iter().for_each(|item| {
            let prompt = format_prompt(&item.instruction);
            let response = format!("{}{}", item.response, END_TOKEN);

            let mut prompt = tokenizer.encode(&prompt);
            let response_ids = tokenizer.encode(&response);

            let mut response = vec![IGNORE_TOKEN; prompt.len()];
            prompt.extend(response_ids.clone());
            response.extend(response_ids);

            prompt.pop();
            response.remove(0);

            if prompt.len() <= context_size {
                let pad_len = context_size - prompt.len();
                prompt.extend(vec![0; pad_len]);
                response.extend(vec![IGNORE_TOKEN; pad_len]);
            } else {
                prompt.truncate(context_size);
                response.truncate(context_size);
            }

            assert_eq!(prompt.len(), response.len());
            assert_eq!(prompt.len(), context_size);

            tokens.push(prompt);
            labels.push(response);
        });

        CodeBotSFTDataset { tokens, labels }
    }
}

impl Dataset<CodeBotItem> for CodeBotSFTDataset {
    fn get(&self, index: usize) -> Option<CodeBotItem> {
        if index >= self.len() {
            return None;
        }

        Some(CodeBotItem {
            tokens: self.tokens[index].clone(),
            labels: self.labels[index].clone(),
        })
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }
}

#[allow(dead_code)]
pub fn sft() {
    let rule_path = Path::new("./data/merge_rules.cbor");
    let sft_path = Path::new("./data/tiny_codes_sft.json");

    let tokenizer = BPETokenizer::load_from(rule_path, END_TOKEN).expect("load tokenizer failed");

    type TrainBackend = Autodiff<Cuda>;

    let device: CudaDevice = Default::default();
    let context_len = 256;
    let batch_size: usize = 32;
    let learning_rate = 3e-4;
    let iterations = 1500;
    let num_epochs = 5;

    let config =
        model::GPTConfig::load("./artifacts/ch02/config.json").expect("config file not found");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load("./artifacts/ch02/final_model".into(), &device)
        .expect("training model not found");

    let model = config.init::<TrainBackend>(&device).load_record(record);

    let sft_text = fs::read_to_string(sft_path).expect("sft path not found");
    let dataset = CodeBotSFTDataset::new(context_len, &sft_text, &tokenizer);

    let max_items = batch_size
        .saturating_mul(iterations)
        .min(dataset.len())
        .min(dataset.len());
    let dataloader_train =
        DataLoaderBuilder::<TrainBackend, CodeBotItem, CodeBotBatch<TrainBackend>>::new(
            CodeBotBatcher,
        )
        .batch_size(batch_size)
        .shuffle(123)
        .set_device(device.clone())
        .build(dataset.clone())
        .slice(0, max_items);

    let dataloader_valid =
        DataLoaderBuilder::<Cuda, CodeBotItem, CodeBotBatch<Cuda>>::new(CodeBotBatcher)
            .batch_size(batch_size)
            .set_device(device.clone())
            .build(dataset)
            .slice(0, max_items);

    let optimizer = AdamWConfig::new().init::<TrainBackend, model::GPT<TrainBackend>>();
    let learner = Learner::new(model, optimizer, learning_rate);

    let result =
        SupervisedTraining::new("./artifacts/ch02_sft", dataloader_train, dataloader_valid)
            .num_epochs(num_epochs)
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .launch(learner);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    result
        .model
        .save_file("./artifacts/ch02/model_sft", &recorder)
        .expect("failed to save the trained model");
}

pub fn chat(
    tokenizer: &crate::tokenizer::BPETokenizer,
    model_path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
) -> String {
    type TrainBackend = Autodiff<Cuda>;

    let device: CudaDevice = Default::default();

    let config =
        model::GPTConfig::load("./artifacts/ch02/config.json").expect("config file not found");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("training model not found");

    let model = config.init::<TrainBackend>(&device).load_record(record);
    let mut tokens = tokenizer.encode(&format_prompt(prompt));

    for _ in 0..max_new_tokens {
        let x = {
            if tokens.len() <= config.max_context_len {
                tokens.clone()
            } else {
                tokens[tokens.len() - config.max_context_len..].to_vec()
            }
        };

        let item = CodeBotItem {
            tokens: x.clone(),
            labels: x,
        };

        let batcher = CodeBotBatcher::default();
        let batch = batcher.batch(vec![item], &device);
        let logits = model.forward(batch.tokens);

        let logits = logits.slice(s![.., -1, ..]).reshape([config.vocab_size]);

        let next_id: TokenID = {
            if temperature == 0.0 {
                logits.argmax(0).into_scalar() as TokenID
            } else {
                let probs = softmax(logits / temperature, 0);
                probs.categorical(1).into_scalar() as TokenID
            }
        };

        if next_id == tokenizer.end_token_id {
            break;
        }

        tokens.push(next_id);
    }

    let mut response = tokenizer.decode(&tokens);
    let response_text = "### Response:";
    if let Some(pos) = response.find(response_text) {
        let substr = &response[(pos + response_text.len())..];
        response = substr.to_string();
    }

    response
}
