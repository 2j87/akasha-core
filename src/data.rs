use crate::tokenizer::AkashaTokenizer;

/// wikitext-103-raw escapes punctuation that the original wiki markup
/// stripper couldn't otherwise represent unambiguously (" @-@ " for a
/// hyphen, " @.@ "/" @,@ " for a period/comma inside a number). Left as-is,
/// the model learns these as real vocabulary/tokens rather than the
/// punctuation they stand in for, observed directly in generated chat output
/// (e.g. "the anti @-@ Byzantine conflict"). Strip them back to plain
/// punctuation before tokenizing.
fn clean_wikitext_markup(text: &str) -> String {
    text.replace(" @-@ ", "-")
        .replace(" @.@ ", ".")
        .replace(" @,@ ", ",")
}

pub struct Dataset {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl Dataset {
    /// Tokenizing the full file in one `encode()` call on the whole text
    /// blew up to a ~14GB allocation (the `tokenizers` crate's working-set
    /// overhead is much larger than the input text size). Reads and
    /// tokenizes the file in fixed-size byte chunks instead, so peak memory
    /// is bounded by `CHUNK_SIZE` regardless of total file size -- this is
    /// what makes it safe to point this at the full ~540MB wikitext-103
    /// training split (or larger) without truncating it.
    pub fn from_file(path: &str, tokenizer: &AkashaTokenizer, seq_len: usize) -> Self {
        use std::io::Read;
        const CHUNK_SIZE: usize = 5_000_000;

        let mut file = std::fs::File::open(path).expect("Cannot open dataset");
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut leftover: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; CHUNK_SIZE];

        loop {
            let n = file.read(&mut buf).expect("Cannot read dataset");
            if n == 0 {
                break;
            }
            leftover.extend_from_slice(&buf[..n]);

            // A fixed byte boundary can land mid-multi-byte UTF-8 sequence
            // (wikitext-103 has scattered non-ASCII chars). Rather than
            // hand-rolling continuation-byte detection (which only catches
            // a sequence cut after its leading byte, not the equally common
            // case of the chunk ending exactly AT the leading byte with zero
            // continuation bytes yet read), defer to `str::from_utf8`'s own
            // error reporting: on failure with no `error_len` (incomplete
            // trailing sequence, not an actually-invalid byte), `valid_up_to`
            // is exactly the safe cut point. Bytes after it are carried into
            // the next chunk instead of corrupting/dropping them.
            let boundary = match std::str::from_utf8(&leftover) {
                Ok(_) => leftover.len(),
                Err(e) => e.valid_up_to(),
            };

            let text = std::str::from_utf8(&leftover[..boundary])
                .expect("invalid UTF-8 in dataset file");
            let cleaned = clean_wikitext_markup(text);
            let chunk_tokens = tokenizer.encode(&cleaned);
            all_tokens.extend_from_slice(&chunk_tokens);
            eprintln!("Tokenized chunk: {} tokens so far", all_tokens.len());

            leftover.drain(..boundary);
        }

        if !leftover.is_empty() {
            if let Ok(text) = std::str::from_utf8(&leftover) {
                all_tokens.extend_from_slice(&tokenizer.encode(&clean_wikitext_markup(text)));
            }
        }

        println!("Dataset: {} tokens total", all_tokens.len());
        Self { tokens: all_tokens, seq_len }
    }

    pub fn random_batch(
        &self,
        batch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);
        for _ in 0..batch_size {
            let max_start = self.tokens.len() - self.seq_len - 1;
            let start = rng.gen_range(0..max_start);
            inputs.extend_from_slice(&self.tokens[start..start + self.seq_len]);
            targets.extend_from_slice(&self.tokens[start + 1..start + self.seq_len + 1]);
        }
        (inputs, targets)
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}
