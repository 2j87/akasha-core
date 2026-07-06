// One-off check: tokenizes a text file using the same chunked pipeline as
// Dataset::from_file (without needing a GPU context) and reports the real
// token count, to verify a corpus actually hits a target token budget
// before committing to a full training run against it.
use akasha_core::data::Dataset;
use akasha_core::tokenizer::AkashaTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("data\\train_combined.txt");
    let tokenizer = AkashaTokenizer::from_pretrained();
    let dataset = Dataset::from_file(path, &tokenizer, 512);
    println!("FINAL TOKEN COUNT: {}", dataset.token_count());
}
