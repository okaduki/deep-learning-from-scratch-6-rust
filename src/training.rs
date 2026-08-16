use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
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
use indicatif::{ProgressBar, ProgressStyle};

use crate::model::{self, GPT, GRPOTrainer, SFTTrainer};
use std::{fmt::format, fs::File, path::Path, vec};

use fancy_regex::Regex;

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

    let gpt_model = config.init::<TrainBackend>(&device);
    let model = model::SFTTrainer::new(gpt_model);

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

    let optimizer = AdamWConfig::new().init::<TrainBackend, model::SFTTrainer<TrainBackend>>();
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

fn generate_<B: Backend>(
    tokenizer: &crate::tokenizer::BPETokenizer,
    model: &GPT<B>,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    max_context_len: usize,
) -> String {
    type TrainBackend = Autodiff<Cuda>;

    let device = &model.devices()[0];

    // let device: CudaDevice = Default::default();
    let mut tokens = tokenizer.encode(prompt);

    for _ in 0..max_new_tokens {
        let x = {
            if tokens.len() <= max_context_len {
                tokens.clone()
            } else {
                tokens[tokens.len() - max_context_len..].to_vec()
            }
        };

        let item = CodeBotItem {
            tokens: x.clone(),
            labels: x,
        };

        let batcher = CodeBotBatcher::default();
        let batch = batcher.batch(vec![item], device);
        let logits = model.forward(batch.tokens);
        let vocab_size = logits.dims()[2];

        let logits = logits.slice(s![.., -1, ..]).reshape([vocab_size]);

        let next_id: TokenID = {
            if temperature == 0.0 {
                logits.argmax(0).into_scalar().elem::<TokenID>()
            } else {
                let probs = softmax(logits / temperature, 0);
                probs.categorical(1).into_scalar().elem::<TokenID>()
            }
        };

        if next_id == tokenizer.end_token_id {
            break;
        }

        tokens.push(next_id);
    }

    tokenizer.decode(&tokens)
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

    let model = SFTTrainer::new(config.init::<TrainBackend>(&device)).load_record(record);
    let model = model.valid();

    generate_(
        tokenizer,
        &model.policy,
        prompt,
        max_new_tokens,
        temperature,
        config.max_context_len,
    )
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
    let target_updates: usize = 1500;

    let config =
        model::GPTConfig::load("./artifacts/ch02/config.json").expect("config file not found");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load("./artifacts/ch02/final_model".into(), &device)
        .expect("training model not found");

    let gpt_model = SFTTrainer::new(config.init::<TrainBackend>(&device))
        .load_record(record)
        .policy;
    let model = model::SFTTrainer::new(gpt_model);

    let sft_text = fs::read_to_string(sft_path).expect("sft path not found");
    let dataset = CodeBotSFTDataset::new(context_len, &sft_text, &tokenizer);

    let max_items = dataset.len();
    let batches_per_epoch = max_items.div_ceil(batch_size);
    let num_epochs = target_updates.div_ceil(batches_per_epoch);
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

    let optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-2)
        .init::<TrainBackend, model::SFTTrainer<TrainBackend>>();
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

    let model = SFTTrainer::new(config.init::<TrainBackend>(&device)).load_record(record);
    let model = model.valid();
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
        let logits = model.policy.forward(batch.tokens);

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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CodeBotGRPOItem {
    prompt: String,
    gt: i32,
    ids: Vec<TokenID>,
    mask: Vec<TokenID>,
}

#[derive(Clone)]
struct CodeBotGRPODataset {
    prompts: Vec<String>,
    gts: Vec<i32>,
    idss: Vec<Vec<TokenID>>,
    masks: Vec<Vec<TokenID>>,
}

impl CodeBotGRPODataset {
    fn new(tokenizer: &BPETokenizer) -> Self {
        let mut prompts = vec![];
        let mut gts = vec![];
        let mut idss = vec![];
        let mut masks = vec![];
        let mut max_len = 0;

        for i in 1..10 {
            for j in 1..10 {
                let instruction = format!("{}+{}=", i, j);
                let prompt = format_prompt(&instruction);
                let response = (i + j).to_string();

                let mut prompt_ids = tokenizer.encode(&prompt);
                let response_ids = tokenizer.encode(&response);
                let mask = [vec![0; prompt_ids.len()], vec![1; response_ids.len()]].concat();

                prompt_ids.extend(response_ids);
                max_len = max_len.max(prompt_ids.len());

                prompts.push(prompt);
                gts.push(i + j);
                idss.push(prompt_ids);
                masks.push(mask);
            }
        }

        for i in 0..idss.len() {
            let pad_len = max_len - idss[i].len();
            idss[i].extend(vec![0; pad_len]);
            masks[i].extend(vec![0; pad_len]);
        }

        CodeBotGRPODataset {
            prompts,
            gts,
            idss,
            masks,
        }
    }
}

impl Dataset<CodeBotGRPOItem> for CodeBotGRPODataset {
    fn get(&self, index: usize) -> Option<CodeBotGRPOItem> {
        if index >= self.len() {
            return None;
        }

        Some(CodeBotGRPOItem {
            prompt: self.prompts[index].clone(),
            gt: self.gts[index],
            ids: self.idss[index].clone(),
            mask: self.masks[index].clone(),
        })
    }

    fn len(&self) -> usize {
        self.idss.len()
    }
}

#[derive(Debug, Clone)]
pub struct CodeBotGRPOBatch<B: Backend> {
    prompts: Vec<String>,
    gts: Vec<i32>,
    ids: Tensor<B, 2, Int>,
    masks: Tensor<B, 2, Float>,
}

#[derive(Clone, Default)]
pub struct CodeBotGRPOBatcher;

impl<B: Backend> Batcher<B, CodeBotGRPOItem, CodeBotGRPOBatch<B>> for CodeBotGRPOBatcher {
    fn batch(&self, items: Vec<CodeBotGRPOItem>, device: &Device<B>) -> CodeBotGRPOBatch<B> {
        assert!(!items.is_empty(), "cannot create an empty batch");

        let context_size = items[0].ids.len();
        assert!(
            items
                .iter()
                .all(|item| item.ids.len() == context_size && item.mask.len() == context_size)
        );

        let ids: Vec<i32> = items
            .iter()
            .flat_map(|item| item.ids.iter().map(|&token| token as i32))
            .collect();
        let masks: Vec<f32> = items
            .iter()
            .flat_map(|item| item.mask.iter().map(|&label| label as f32))
            .collect();

        CodeBotGRPOBatch {
            prompts: items.iter().map(|i| i.prompt.clone()).collect(),
            gts: items.iter().map(|i| i.gt).collect(),
            ids: Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device)
                .reshape([items.len(), context_size]),
            masks: Tensor::<B, 1, Float>::from_floats(masks.as_slice(), device)
                .reshape([items.len(), context_size]),
        }
    }
}

fn calculate_reward(ground_truth: i32, response: &str) -> f32 {
    let re = Regex::new(r"(-?\d+)").unwrap();

    re.find_iter(response).last().map_or(0.0, |match_res| {
        if let Ok(m) = match_res {
            let s = m.as_str();
            s.parse()
                .map_or(0.0, |d: i32| if d == ground_truth { 1.0 } else { 0.0 })
        } else {
            0.0
        }
    })
}

#[derive(Debug, Clone)]
struct GrpoSampleLog {
    prompt: String,
    response: String,
    reward: f32,
    advantage: f32,
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let text = text.replace('\n', "\\n");
    let mut preview: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        preview.push_str("...");
    }
    preview
}

fn generate_group<B: Backend>(
    model: &GRPOTrainer<Cuda>,
    tokenizer: &crate::tokenizer::BPETokenizer,
    items: &CodeBotGRPOBatch<B>,
    group_size: usize,
    max_context_len: usize,
) -> (
    Vec<String>,
    Vec<String>,
    CodeBotGRPOBatch<B>,
    Tensor<B, 2>,
    Vec<GrpoSampleLog>,
) {
    let device = &items.ids.device();
    let mut all_prompts = vec![];
    let mut all_responses = vec![];
    let mut generated_ids = vec![];
    let mut generated_masks = vec![];
    let mut all_advantages = vec![];
    let mut sample_logs = vec![];
    let mut max_len = 0;

    for i in 0..items.gts.len() {
        let prompt = &items.prompts[i];
        let gt = items.gts[i];

        let mut rewards = vec![];
        let mut group_responses = vec![];

        for _ in 0..group_size {
            let max_new_tokens = 100;
            let temperature = 1.0;
            let full_text = generate_(
                tokenizer,
                &model.policy,
                prompt,
                max_new_tokens,
                temperature,
                max_context_len,
            );
            let response = full_text
                .strip_prefix(prompt)
                .unwrap_or(&full_text)
                .to_string();
            let reward = calculate_reward(gt, &response);

            all_prompts.push(prompt.to_string());
            all_responses.push(response.clone());
            group_responses.push(response);
            rewards.push(reward);
        }

        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        for (response, reward) in group_responses.into_iter().zip(rewards) {
            let advantage = reward - mean_reward;
            let prompt_ids = tokenizer.encode(prompt);
            let response_ids = tokenizer.encode(&response);
            let mut ids = [prompt_ids.clone(), response_ids.clone()].concat();
            let mut mask = [vec![0; prompt_ids.len()], vec![1; response_ids.len()]].concat();

            if ids.len() > max_context_len {
                let start = ids.len() - max_context_len;
                ids = ids[start..].to_vec();
                mask = mask[start..].to_vec();
            }

            max_len = max_len.max(ids.len());
            generated_ids.push(ids);
            generated_masks.push(mask);
            all_advantages.push(advantage);
            sample_logs.push(GrpoSampleLog {
                prompt: prompt.clone(),
                response,
                reward,
                advantage,
            });
        }
    }

    for i in 0..generated_ids.len() {
        let pad_len = max_len - generated_ids[i].len();
        generated_ids[i].extend(vec![0; pad_len]);
        generated_masks[i].extend(vec![0; pad_len]);
    }

    let flat_ids: Vec<i32> = generated_ids
        .iter()
        .flat_map(|ids| ids.iter().map(|&token| token as i32))
        .collect();
    let flat_masks: Vec<f32> = generated_masks
        .iter()
        .flat_map(|mask| mask.iter().map(|&value| value as f32))
        .collect();
    let generated_batch = CodeBotGRPOBatch {
        prompts: all_prompts.clone(),
        gts: vec![0; generated_ids.len()],
        ids: Tensor::<B, 1, Int>::from_ints(flat_ids.as_slice(), device)
            .reshape([generated_ids.len(), max_len]),
        masks: Tensor::<B, 1, Float>::from_floats(flat_masks.as_slice(), device)
            .reshape([generated_ids.len(), max_len]),
    };
    let advantages = Tensor::<B, 1, Float>::from_floats(all_advantages.as_slice(), device)
        .reshape([all_advantages.len(), 1]);

    (
        all_prompts,
        all_responses,
        generated_batch,
        advantages,
        sample_logs,
    )
}

fn grpo_loss<B: Backend>(
    model: &GRPOTrainer<B>,
    old_model: &GRPOTrainer<B>,
    items: &CodeBotGRPOBatch<B>,
    advantages: &Tensor<B, 2>,
    epsilon: f32,
) -> Tensor<B, 1> {
    let ids = items.ids.clone();
    let masks = items.masks.clone();
    let n_samples = ids.dims()[0] as f32;

    let compute_probs = |model: &GPT<B>, ids: Tensor<B, 2, Int>| {
        let logits = model.forward(ids.clone());
        let [batch_size, sequence_length, _vocab_size] = logits.dims();

        let tail = logits.slice(s![.., ..-1, ..]);
        let labels = ids
            .slice(s![.., 1..])
            .reshape([batch_size, sequence_length - 1, 1]);
        let probs = softmax(tail, 2);
        let token_probs = probs
            .gather(2, labels)
            .reshape([batch_size, sequence_length - 1]);
        token_probs
    };

    let probs = compute_probs(&model.policy, ids.clone());
    let old_probs = compute_probs(&old_model.policy, ids);

    let ratio = probs / (old_probs + 1e-8);

    let unclipped = ratio.clone() * advantages.clone();
    let clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages.clone();

    let mask = masks.slice(s![.., 1..]);
    let objective = unclipped.min_pair(clipped) * mask;

    let y = -objective.sum() / n_samples;
    y
}

#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn GRPO() {
    let rule_path = Path::new("./data/merge_rules.cbor");
    let tokenizer = BPETokenizer::load_from(rule_path, END_TOKEN).expect("load tokenizer failed");

    type TrainBackend = Autodiff<Cuda>;

    let device: CudaDevice = Default::default();
    let learning_rate = 7e-6;
    let max_iters: usize = 500;
    let n_update_per_generation = 2;
    let eval_interval = 10;
    let epsilon = 0.2;
    let group_size = 8;
    let batch_size: usize = 32;
    let _context_len = 256;
    let max_new_tokens = 100;

    let config =
        model::GPTConfig::load("./artifacts/ch02/config.json").expect("config file not found");

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load("./artifacts/ch02/model_sft".into(), &device)
        .expect("training model not found");

    let gpt_model = SFTTrainer::new(config.init::<TrainBackend>(&device))
        .load_record(record)
        .policy;
    let mut model = model::GRPOTrainer::new(gpt_model);
    model.set_dropout(0.0);

    let dataset = CodeBotGRPODataset::new(&tokenizer);

    let max_items = dataset.len();
    let dataloader_train =
        DataLoaderBuilder::<TrainBackend, CodeBotGRPOItem, CodeBotGRPOBatch<TrainBackend>>::new(
            CodeBotGRPOBatcher,
        )
        .batch_size(batch_size)
        .shuffle(123)
        .set_device(device.clone())
        .build(dataset.clone())
        .slice(0, max_items);

    let dataloader_valid =
        DataLoaderBuilder::<Cuda, CodeBotGRPOItem, CodeBotGRPOBatch<Cuda>>::new(CodeBotGRPOBatcher)
            .batch_size(1)
            .set_device(device.clone())
            .build(dataset)
            .slice(0, max_items);

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-2)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    // let train_batches: Vec<_> = dataloader_train.iter().collect();
    // assert!(!train_batches.is_empty(), "training dataset is empty");

    let total_batches = max_iters as u64;
    let progress = ProgressBar::new(total_batches);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg} ({eta})")
            .unwrap(),
    );

    let mut i = 0;
    let mut accuracies = vec![];
    while i < max_iters {
        for batch in dataloader_train.iter() {
            if i >= max_iters {
                break;
            }

            // let batch = train_batches[i % train_batches.len()].clone();
            progress.set_message(format!("iter {}: generate", i));
            let old_model = model.clone().no_grad();
            let old_model_valid = old_model.valid();

            let (_all_prompts, _all_responses, generated_batch, all_advantages, sample_logs) =
                generate_group(
                    &old_model_valid,
                    &tokenizer,
                    &batch,
                    group_size,
                    config.max_context_len,
                );

            progress.set_message(format!("iter {}: update", i));
            for _ in 0..n_update_per_generation {
                let loss = grpo_loss(
                    &model,
                    &old_model,
                    &generated_batch,
                    &all_advantages,
                    epsilon,
                );
                let grads = loss.backward();

                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(learning_rate, model, grads);
            }

            if i % eval_interval == 0 {
                progress.set_message(format!("iter {}: valid", i));
                let model_valid = model.valid();
                let mut correct = 0;
                let mut total = 0;
                for batch in dataloader_valid.iter() {
                    let response = generate_(
                        &tokenizer,
                        &model_valid.policy,
                        &batch.prompts[0],
                        max_new_tokens,
                        0.0,
                        config.max_context_len,
                    );
                    let reward = calculate_reward(batch.gts[0], &response);
                    correct += if reward > 0.0 { 1 } else { 0 };
                    total += 1;
                }

                let correct_accuracy = correct as f64 / total as f64;
                accuracies.push(correct_accuracy);

                progress.println(format!(
                    "[Valid - Iteration {}] Accuracy {:.3}({}/{})",
                    i, correct_accuracy, correct, total
                ));
                for (sample_index, sample) in sample_logs.iter().take(10).enumerate() {
                    progress.println(format!(
                        "[Sample {}] reward={:.3} advantage={:.3} prompt=\"{}\" response=\"{}\"",
                        sample_index,
                        sample.reward,
                        sample.advantage,
                        preview_text(&sample.prompt, 80),
                        preview_text(&sample.response, 120),
                    ));
                }
            }

            i += 1;
            progress.inc(1);
        }
    }

    progress.set_message("saving");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("./artifacts/ch02/model_grpo", &recorder)
        .expect("failed to save the trained model");
    progress.finish_with_message("done");
}
