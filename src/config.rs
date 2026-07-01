// ~117M parameters - GPT-2 Small scale
pub const DIM: u32 = 768;
pub const NUM_HEADS: u32 = 12;
pub const HEAD_DIM: u32 = DIM / NUM_HEADS; // 64
pub const NUM_LAYERS: usize = 12;
pub const SEQ_LEN: u32 = 512;
pub const FFN_HIDDEN: u32 = 3072; // 4 x DIM
pub const VOCAB_SIZE: u32 = 50257; // GPT-2 tokenizer

pub const BATCH_SIZE: usize = 2;
pub const ACCUMULATION_STEPS: usize = 32; // effective batch = 64

// 2e-4 was tested for ~3000 post-restart steps (195200-198150) and showed no
// net loss reduction (bucketed averages stayed flat ~4.1-4.25, vs a real,
// if slow, decline from 4.86->4.05 over the prior 150k steps at LR in the
// ~1.2-1.5e-4 range) -- too aggressive at this point in training, just
// adding oscillation without progress. Lowered back into the empirically
// productive range.
pub const LR_MAX: f32 = 1.2e-4;
pub const LR_MIN: f32 = 1e-5;
pub const WARMUP_STEPS: usize = 200;
pub const MAX_STEPS: usize = 500_000;
pub const SAVE_EVERY: usize = 5000;
pub const LOG_EVERY: usize = 50;

pub const ADAM_WEIGHT_DECAY: f32 = 0.01;
pub const GRAD_CLIP_NORM: f32 = 1.0;

// LR warm-restart anchor: the original cosine schedule (MAX_STEPS=200_000)
// had decayed to its LR_MIN floor by the time training plateaued. This
// re-bases the schedule to start a fresh warmup+cosine-decay cycle from the
// actual checkpoint being resumed from (195_000 -- NOT the step number
// training appeared to be stuck at, which was itself a redundant replay of
// the tail end of the already-completed first run; see train.bat's
// auto-restart interacting with find_latest_checkpoint() not recognizing
// model_final.bin). Anchoring to the real resume point avoids a dead
// zero-LR gap that anchoring to the wrong step would otherwise create.
pub const LR_RESTART_STEP: usize = 195_000;

pub fn cosine_lr(step: usize, warmup_steps: usize, max_steps: usize, lr_max: f32, lr_min: f32) -> f32 {
    let step = step.saturating_sub(LR_RESTART_STEP);
    let max_steps = max_steps.saturating_sub(LR_RESTART_STEP);
    if step < warmup_steps {
        return lr_max * step as f32 / warmup_steps as f32;
    }
    let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}
