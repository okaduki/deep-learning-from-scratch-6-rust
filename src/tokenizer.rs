use std::{fs::File, path::Path};

use fancy_regex::Regex;
use indexmap::IndexMap;
use indicatif::ProgressBar;

use ciborium::de::from_reader;

type TokenID = u32;

pub struct BPETokenizer {
    merge_rules: IndexMap<(TokenID, TokenID), TokenID>,
    id_to_bytes: IndexMap<TokenID, Vec<u8>>,
    vocab_size: usize,
    end_token: String,
    end_token_id: TokenID,
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
        let texts: Vec<_> = text.split(&self.end_token).collect();

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

        for (i, sentence) in texts.iter().enumerate() {
            if let Some(pb) = &pb_opt {
                pb.inc(1);
            }

            if i > 0 {
                ids.push(self.end_token_id);
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

fn count_pairs(ids: &[TokenID], counts: &mut IndexMap<(TokenID, TokenID), usize>) {
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
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

pub fn train_bpe(text: &str, vocab_size: usize) -> IndexMap<(TokenID, TokenID), TokenID> {
    if vocab_size <= 257 {
        return IndexMap::new();
    }

    let mut ids_list: Vec<_> = text
        .split("<|endoftext|>")
        .flat_map(|x| pretokenize(x))
        .map(encode_text)
        .collect();

    let num_merges = vocab_size - 256 - 1;
    let mut merged_rules = IndexMap::new();

    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} it ({per_sec})")
            .unwrap(),
    );

    for step in 0..num_merges {
        pb.inc(1);

        let mut counts = IndexMap::new();

        for ids in &mut ids_list {
            count_pairs(&ids, &mut counts);
        }

        let best_pair = counts.iter().max_by_key(|(k, v)| (**v, **k));
        if let Some((pair, _)) = best_pair {
            let new_id = (step + 256) as TokenID;
            merged_rules.insert(*pair, new_id);

            for ids in &mut ids_list {
                *ids = merge(&ids, *pair, new_id);
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

        count_pairs(&ids, &mut counts);
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
