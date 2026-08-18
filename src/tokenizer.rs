use std::{
    fs::File,
    io::{Read, Seek},
    ops::Index,
    os::unix::fs::FileExt,
    path::Path,
};

use fancy_regex::{Input, Regex};
use indexmap::IndexMap;
use indexmap::IndexSet;
use indicatif::ProgressBar;

use ciborium::de::from_reader;

use rayon::prelude::*;

pub type TokenID = u16;

pub struct BPETokenizer {
    merge_rules: IndexMap<(TokenID, TokenID), TokenID>,
    id_to_bytes: IndexMap<TokenID, Vec<u8>>,
    vocab_size: usize,
    end_token: String,
    pub end_token_id: TokenID,
    show_progress: bool,
}

pub const END_TOKEN: &str = "<|endoftext|>";

impl BPETokenizer {
    pub fn new(merge_rules: &IndexMap<(TokenID, TokenID), TokenID>, end_token: &str) -> Self {
        let mut id_to_bytes: IndexMap<TokenID, Vec<u8>> = IndexMap::new();

        for i in 0..256 {
            id_to_bytes.insert(i, vec![i as u8]);
        }
        let end_token_id = (256 + merge_rules.len()) as TokenID;

        for (&(p1, p2), &id) in merge_rules {
            let mut bytes1 = id_to_bytes.get(&p1).unwrap().clone();
            let mut bytes2 = id_to_bytes.get(&p2).unwrap().clone();
            bytes1.append(&mut bytes2);
            id_to_bytes.insert(id, bytes1);
        }
        id_to_bytes.insert(end_token_id, end_token.as_bytes().to_vec());

        let vocab_size = id_to_bytes.len();
        BPETokenizer {
            merge_rules: merge_rules.clone(),
            id_to_bytes,
            vocab_size,
            end_token: end_token.to_string(),
            end_token_id,
            show_progress: false,
        }
    }

    pub fn load_from(path: &Path, end_token: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let merge_rules: IndexMap<(TokenID, TokenID), TokenID> = from_reader(file)?;

        Ok(BPETokenizer::new(&merge_rules, end_token))
    }

    pub fn encode(&self, text: &str) -> Vec<TokenID> {
        let mut ids = vec![];
        let texts: Vec<_> = text.split_inclusive(&self.end_token).collect();

        let pb_opt = if self.show_progress {
            Some(ProgressBar::new(texts.len() as u64))
        } else {
            None
        };

        if let Some(pb) = &pb_opt {
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40} {pos}/{len} it ({per_sec})")
                    .unwrap(),
            );
        }

        for sentence in texts {
            if let Some(pb) = &pb_opt {
                pb.inc(1);
            }

            if let Some(sentence) = sentence.strip_suffix(&self.end_token) {
                for pretoken in pretokenize(sentence) {
                    let mut tokens = encode_text(pretoken);

                    for (&pair, &new_id) in &self.merge_rules {
                        tokens = merge(&tokens, pair, new_id);
                    }

                    ids.append(&mut tokens);
                }
                ids.push(self.end_token_id);
                continue;
            }

            for pretoken in pretokenize(sentence) {
                let mut tokens = encode_text(pretoken);

                for (&pair, &new_id) in &self.merge_rules {
                    tokens = merge(&tokens, pair, new_id);
                }

                ids.append(&mut tokens);
            }
        }
        if let Some(pb) = &pb_opt {
            pb.finish();
        }

        ids
    }

    pub fn decode(&self, ids: &[TokenID]) -> String {
        let ids: Vec<u8> = ids
            .iter()
            .map(|id| self.id_to_bytes.get(id).unwrap())
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&ids).to_string()
    }
}

fn encode_text(text: &str) -> Vec<TokenID> {
    text.as_bytes().iter().map(|&x| x as TokenID).collect()
}

fn pretokenize(text: &str) -> Vec<&str> {
    let re = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap();

    re.find_iter(text).map(|x| x.unwrap().as_str()).collect()
}

fn count_pairs(ids: &[TokenID], counts: &mut IndexMap<(TokenID, TokenID), usize>, weight: usize) {
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += weight;
    }
}

fn merge(ids: &[TokenID], pair: (TokenID, TokenID), new_id: TokenID) -> Vec<TokenID> {
    let mut merged = Vec::new();

    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            merged.push(new_id);
            i += 2;
        } else {
            merged.push(ids[i]);
            i += 1;
        }
    }

    merged
}

pub fn train_bpe(text_path: &Path, vocab_size: usize) -> IndexMap<(TokenID, TokenID), TokenID> {
    if vocab_size <= 257 {
        return IndexMap::new();
    }

    let chunk_pos = {
        let mut input_file = File::open(text_path).expect("file not found.");

        const CHUNK_SIZE: usize = 4096;
        let mut buffer = [0; CHUNK_SIZE];
        let end_token = END_TOKEN.as_bytes();
        let end_token_len = end_token.len();
        let mut chunk_pos = vec![0];

        loop {
            let offset = input_file.stream_position().unwrap() as usize;
            let n = input_file.read(&mut buffer).unwrap();
            if n == 0 {
                break;
            }

            let chunk = &buffer[..n];
            if let Some(pos) = chunk.windows(end_token_len).position(|w| w == end_token) {
                let next_pos = offset + pos + end_token_len;
                chunk_pos.push(next_pos);
                input_file
                    .seek(std::io::SeekFrom::Start(next_pos as u64))
                    .unwrap();
            } else if n == CHUNK_SIZE {
                input_file
                    .seek(std::io::SeekFrom::Current(-(end_token_len as i64)))
                    .unwrap();
            } else {
                chunk_pos.push(input_file.metadata().unwrap().len() as usize);
                break;
            }
        }

        chunk_pos
    };

    let mut pretoken_counts: IndexMap<String, usize> = IndexMap::new();
    {
        let ranges: Vec<(usize, usize)> = chunk_pos.windows(2).map(|sl| (sl[0], sl[1])).collect();
        let pb = ProgressBar::new(ranges.len() as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len}({percent}%), {per_sec} ({eta})")
                .unwrap(),
        );
        pb.set_message(format!("pretoken"));

        let re =
            Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
                .unwrap();

        let task_count = rayon::current_num_threads() * 8;
        let ranges_per_task = ranges.len().div_ceil(task_count).max(1);

        let pretoken_countss: Vec<IndexMap<String, usize>> = ranges
            .par_chunks(ranges_per_task)
            .map(|group| {
                let mut pretoken_counts: IndexMap<String, usize> = IndexMap::new();
                let input_file = File::open(text_path).expect("file not found.");
                let mut buf = Vec::new();

                for &(begin, end) in group {
                    buf.resize(end - begin, 0);

                    input_file
                        .read_exact_at(&mut buf, begin as u64)
                        .expect("seek error");

                    let text = String::from_utf8_lossy(&buf).to_string();
                    let pretokens: Vec<&str> =
                        re.find_iter(&text).map(|x| x.unwrap().as_str()).collect();
                    //  let pretokens = pretokenize(&text);

                    pretokens.iter().for_each(|pretoken| {
                        *pretoken_counts.entry(pretoken.to_string()).or_insert(0) += 1;
                    });

                    pb.inc(1);
                }

                pretoken_counts
            })
            .collect();
        pb.finish();

        for local_counts in pretoken_countss {
            for (token, count) in local_counts {
                *pretoken_counts.entry(token).or_insert(0) += count;
            }
        }
    }

    let mut ids_counts: IndexMap<Vec<TokenID>, usize> = IndexMap::new();
    pretoken_counts.iter().for_each(|(pretoken, &count)| {
        ids_counts.insert(encode_text(pretoken), count);
    });

    let num_merges = vocab_size - 256 - 1;
    let mut merged_rules = IndexMap::new();

    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} it ({per_sec})")
            .unwrap(),
    );

    let mut pair_to_ids: IndexMap<(TokenID, TokenID), IndexSet<Vec<TokenID>>> = IndexMap::new();
    let mut pair_counts = IndexMap::new();
    for (ids, weight) in &ids_counts {
        count_pairs(&ids, &mut pair_counts, *weight);

        for sl in ids.windows(2) {
            let pair = (sl[0], sl[1]);
            pair_to_ids
                .entry(pair)
                .or_insert(IndexSet::new())
                .insert(ids.clone());
        }
    }

    for step in 0..num_merges {
        pb.inc(1);

        let best_pair = pair_counts.iter().max_by_key(|(k, v)| (**v, **k));
        if let Some((&best_pair, _)) = best_pair {
            let new_id = (step + 256) as TokenID;
            merged_rules.insert(best_pair, new_id);

            if let Some(affected_ids) = pair_to_ids.get(&best_pair) {
                let affected_ids = affected_ids.clone();
                pair_to_ids.swap_remove(&best_pair);

                for ids in affected_ids.iter() {
                    let new_ids = merge(ids, best_pair, new_id);
                    let ids_count = *ids_counts.get(ids).unwrap_or(&0);

                    ids_counts.swap_remove(ids);
                    *ids_counts.entry(new_ids.clone()).or_insert(0) += ids_count;

                    let mut old_counts = IndexMap::new();
                    count_pairs(&ids, &mut old_counts, 1);

                    for (pair, count) in old_counts {
                        pair_counts[&pair] -= count * ids_count;
                        if pair_counts[&pair] <= 0 {
                            pair_counts.swap_remove(&pair);
                        }
                        pair_to_ids.entry(pair).and_modify(|set| {
                            set.swap_remove(ids);
                        });
                    }

                    let mut new_counts = IndexMap::new();
                    count_pairs(&new_ids, &mut new_counts, 1);
                    for (pair, count) in new_counts {
                        *pair_counts.entry(pair).or_insert(0) += count * ids_count;
                        pair_to_ids.entry(pair).or_default().insert(new_ids.clone());
                    }
                }
            }
        } else {
            break;
        }
    }
    pb.finish();

    merged_rules
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::*;

    #[test]
    fn test_count_pairs() {
        let ids = [1, 2, 3, 1, 2];
        let mut counts = IndexMap::new();

        count_pairs(&ids, &mut counts, 1);
        assert_eq!(
            counts,
            IndexMap::from([((1, 2), 2), ((2, 3), 1), ((3, 1), 1),])
        );
    }

    #[test]
    fn test_merge() {
        let ids = [1, 2, 3, 1, 2];
        let pair = (1, 2);
        let new_id = 4;
        let expected = [4, 3, 4];

        let merged = merge(&ids, pair, new_id);
        assert_eq!(merged, expected);
    }

    #[test]
    fn test_pretokenize() {
        let text = "Hello! I'm fine.";
        let actual = pretokenize(text);

        assert_eq!(actual, ["Hello", "!", " I", "'m", " fine", ".",]);
    }

    #[test]
    fn test_encode_decode() {
        let merge_rules: IndexMap<(TokenID, TokenID), TokenID> = IndexMap::from([
            ((105, 115), 256),
            ((256, 32), 257),
            ((105, 110), 258),
            ((72, 101), 259),
        ]);

        let text = "Hello世界😁";
        let tokenizer = BPETokenizer::new(&merge_rules, END_TOKEN);
        let ids = tokenizer.encode(text);

        let expected = vec![
            259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129,
        ];
        assert_eq!(ids, expected);

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_split_encode_decode() {
        let merge_rules: IndexMap<(TokenID, TokenID), TokenID> = IndexMap::from([]);

        let text = "a<|endoftext|>b<|endoftext|>c<|endoftext|>d<|endoftext|>";
        let tokenizer = BPETokenizer::new(&merge_rules, END_TOKEN);
        let ids = tokenizer.encode(text);

        let expected = vec![97, 256, 98, 256, 99, 256, 100, 256];
        assert_eq!(ids, expected);

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }
}
