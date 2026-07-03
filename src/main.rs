use std::sync::Arc;

use akasha_core::config::*;
use akasha_core::data::Dataset;
use akasha_core::nn::akasha_model::AkashaModel;
use akasha_core::tokenizer::AkashaTokenizer;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

fn select_backend() -> WgpuContext {
    #[cfg(feature = "cuda")]
    {
        if let Some(ctx) = WgpuContext::new_cuda() {
            return ctx;
        }
    }
    println!("[wilupgu] Vulkan backend selected");
    pollster::block_on(WgpuContext::new())
}

fn find_latest_checkpoint(dir: &str) -> Option<(String, usize)> {
    std::fs::read_dir(dir).ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            if name == "model_final.bin" {
                // model_final.bin is saved once the training loop finishes
                // (step == MAX_STEPS). Without recognizing it here,
                // find_latest_checkpoint only ever sees model_step_*.bin
                // files, so a completed run that gets auto-restarted (e.g.
                // by train.bat's wrapper loop) resumes from the last
                // *periodic* checkpoint instead of the actually-finished
                // model -- silently re-running (and re-saving over) the
                // last SAVE_EVERY steps every time the process restarts.
                return Some((e.path().to_str()?.to_string(), MAX_STEPS));
            }
            let step = name
                .strip_prefix("model_step_")?
                .strip_suffix(".bin")?
                .parse::<usize>().ok()?;
            Some((e.path().to_str()?.to_string(), step))
        })
        .max_by_key(|(_, step)| *step)
}

/// Keeps only the `keep` most recent `model_step_*.bin` checkpoints (deletes
/// older ones). Checkpoints are ~1.3GB each, so unbounded accumulation fills
/// the disk over a long run.
fn cleanup_old_checkpoints(dir: &str, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut checkpoints: Vec<(String, usize)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let step = name
                .strip_prefix("model_step_")?
                .strip_suffix(".bin")?
                .parse::<usize>().ok()?;
            Some((e.path().to_str()?.to_string(), step))
        })
        .collect();
    checkpoints.sort_by_key(|(_, step)| *step);
    if checkpoints.len() > keep {
        for (path, _) in &checkpoints[..checkpoints.len() - keep] {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn build_model(ctx: Arc<WgpuContext>) -> AkashaModel {
    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; SEQ_LEN as usize],
    ));
    AkashaModel::new(ctx, VOCAB_SIZE, DIM, SEQ_LEN, NUM_LAYERS, NUM_HEADS, &input_tokens)
}

fn run_chat(use_latest_checkpoint: bool) {
    let ctx = Arc::new(select_backend());
    let tokenizer = AkashaTokenizer::from_pretrained();
    let model = build_model(ctx);

    let model_path = if use_latest_checkpoint {
        match find_latest_checkpoint("checkpoints") {
            Some((path, step)) => {
                println!("Loading latest checkpoint: {} (step {})", path, step);
                path
            }
            None => {
                println!("No checkpoints found, falling back to model_final.bin");
                "checkpoints\\model_final.bin".to_string()
            }
        }
    } else {
        "checkpoints\\model_final.bin".to_string()
    };

    model
        .load_weights(&model_path)
        .unwrap_or_else(|_| panic!("Failed to load {} -- train the model first", model_path));

    println!("Model loaded. Type a prompt (Ctrl+C to exit):\n");
    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let tokens = tokenizer.encode(input);
        let output = model.generate(&tokenizer, &tokens, 200, 0.8);
        println!("{}\n", output);
    }
}

fn run_training() {
    let backend = select_backend();
    let ctx = Arc::new(backend);

    let tokenizer = AkashaTokenizer::from_pretrained();
    println!("Vocab size: {}", tokenizer.vocab_size());

    let dataset = Dataset::from_file("data\\train_combined.txt", &tokenizer, SEQ_LEN as usize);
    println!("Dataset: {} tokens", dataset.token_count());

    let model = build_model(ctx);
    println!("Model ready - ~117M parameters (12-head attention)");

    std::fs::create_dir_all("checkpoints").unwrap();

    let start_step = match find_latest_checkpoint("checkpoints") {
        Some((path, step)) => {
            model.load_weights(&path).expect("Failed to load checkpoint");
            println!("Resumed from: {} (step {})", path, step);
            step + 1
        }
        None => {
            println!("Starting fresh training run");
            0
        }
    };

    if start_step >= MAX_STEPS {
        // Training already reached MAX_STEPS in a prior run (model_final.bin
        // exists and is recognized as the latest checkpoint). Without this
        // guard, `for step in start_step..MAX_STEPS` is an empty range, the
        // loop body (and therefore every loss update) never runs, and the
        // code below unconditionally re-saves model_final.bin and prints
        // "Best loss: f32::MAX" -- then train.bat's wrapper loop restarts
        // the process immediately, which repeats this instantly forever
        // (observed as a multi-second burst of hundreds of "Starting/
        // resuming" log lines with no actual training happening).
        println!(
            "Training already complete: resumed step {} >= MAX_STEPS {}. Nothing to do.",
            start_step, MAX_STEPS
        );
        println!("Increase MAX_STEPS in config.rs to continue training further.");
        return;
    }

    let mut rng = rand::thread_rng();
    let mut best_loss = f32::MAX;
    // Raw per-step loss is extremely noisy (BATCH_SIZE=2 -> ~1024 tokens per
    // reading, sampled from a now-diverse multi-genre corpus), which makes it
    // easy to misjudge a real (if slow) trend as a "stuck" plateau just from
    // eyeballing individual printed values. An EMA (alpha=0.99, ~100-step
    // effective window) tracks the actual trend without needing to manually
    // bucket-average the log after the fact.
    let mut ema_loss: Option<f32> = None;
    const EMA_ALPHA: f32 = 0.99;

    println!("Training started.");
    println!("{:>8} | {:>8} | {:>10} | {:>10}", "step", "loss", "ema_loss", "lr");
    println!("{}", "-".repeat(48));

    for step in start_step..MAX_STEPS {
        let lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LR_MAX, LR_MIN);
        let (inputs, targets) = dataset.random_batch(BATCH_SIZE, &mut rng);

        let loss = model.train_step(&inputs, &targets, BATCH_SIZE, lr, step, ACCUMULATION_STEPS);

        if let Some(l) = loss {
            if l < best_loss {
                best_loss = l;
            }
            ema_loss = Some(match ema_loss {
                Some(prev) => EMA_ALPHA * prev + (1.0 - EMA_ALPHA) * l,
                None => l,
            });

            if step % LOG_EVERY == 0 {
                println!(
                    "step {:6} | loss {:.4} | ema_loss {:.4} | lr {:.2e}",
                    step, l, ema_loss.unwrap(), lr
                );
            }

            if l.is_nan() || l.is_infinite() {
                eprintln!("ERROR: Loss is NaN at step {}. Stopping.", step);
                eprintln!("Try reducing LR_MAX to 1e-4 and restart.");
                std::process::exit(1);
            }
        }

        if step % SAVE_EVERY == 0 && step > 0 {
            let path = format!("checkpoints\\model_step_{}.bin", step);
            model.save_weights(&path).unwrap();
            println!("--- Checkpoint saved: {} ---", path);
            cleanup_old_checkpoints("checkpoints", 3);
        }
    }

    model.save_weights("checkpoints\\model_final.bin").unwrap();

    let config_json = format!(
        r#"{{
  "dim": {},
  "num_layers": {},
  "seq_len": {},
  "ffn_hidden": {},
  "vocab_size": {},
  "trained_steps": {},
  "best_loss": {:.4}
}}"#,
        DIM, NUM_LAYERS, SEQ_LEN, FFN_HIDDEN, VOCAB_SIZE, MAX_STEPS, best_loss
    );
    std::fs::write("checkpoints\\config.json", config_json).unwrap();

    println!("Training complete!");
    println!("Best loss: {:.4}", best_loss);
    println!("Model saved: checkpoints\\model_final.bin");
    println!("Run with: cargo run --release --bin akasha-core -- --chat");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--chat-checkpoint") {
        run_chat(true);
    } else if args.iter().any(|a| a == "--chat") {
        run_chat(false);
    } else {
        run_training();
    }
}
