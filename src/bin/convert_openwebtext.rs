// One-off conversion tool: reads Skylion007/openwebtext parquet shards
// (single "text" string column per row/document) and concatenates every
// document into a single plain-text file, one document per line-separated
// block, ready to be appended to data/train_combined.txt. Not part of the
// training hot path -- run once, then discard the parquet shards.
use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_dir = args.get(1).map(|s| s.as_str()).unwrap_or("data\\openwebtext_parquet");
    let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("data\\openwebtext_extracted.txt");

    let mut shard_paths: Vec<_> = std::fs::read_dir(input_dir)
        .expect("cannot read input dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "parquet"))
        .collect();
    shard_paths.sort();

    println!("Found {} parquet shards in {}", shard_paths.len(), input_dir);

    let out_file = File::create(output_path).expect("cannot create output file");
    let mut writer = BufWriter::new(out_file);
    let mut total_docs = 0usize;
    let mut total_bytes = 0u64;

    for path in &shard_paths {
        let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {path:?}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("cannot read parquet metadata for {path:?}: {e}"));
        let reader = builder.build().unwrap_or_else(|e| panic!("cannot build reader for {path:?}: {e}"));

        let mut shard_docs = 0usize;
        for batch_result in reader {
            let batch = batch_result.unwrap_or_else(|e| panic!("error reading batch from {path:?}: {e}"));
            let col = batch
                .column_by_name("text")
                .unwrap_or_else(|| panic!("no 'text' column in {path:?}"));
            let text_array = col
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap_or_else(|| panic!("'text' column in {path:?} is not a StringArray"));

            for i in 0..text_array.len() {
                if text_array.is_null(i) {
                    continue;
                }
                let doc = text_array.value(i);
                writer.write_all(doc.as_bytes()).unwrap();
                writer.write_all(b"\n\n").unwrap();
                total_bytes += doc.len() as u64 + 2;
                shard_docs += 1;
            }
        }
        total_docs += shard_docs;
        println!(
            "  {:?}: {} docs ({:.2} GB running total)",
            path.file_name().unwrap(),
            shard_docs,
            total_bytes as f64 / 1e9
        );
    }

    writer.flush().unwrap();
    println!(
        "Done: {} documents, {:.2} GB written to {}",
        total_docs,
        total_bytes as f64 / 1e9,
        output_path
    );
}
