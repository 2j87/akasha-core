@echo off
set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64
rem Root cause of the earlier CUDA instability found and fixed: Linear::backward's
rem grad_input MatMulTrp call used a meta tensor with swapped N/K (reused the
rem forward meta instead of [m, in_features, out_features]), corrupting the
rem gradient flowing back from lm_head on every step on BOTH backends. Confirmed
rem fixed via diagnose.rs CHECK 8 passing cleanly (loss -> 0.0000) on CUDA. CUDA's
rem cuBLAS GEMMs are also faster than Vulkan/WGSL compute shaders for this
rem workload, so switching back to CUDA for real training.
:loop
echo [%date% %time%] Starting/resuming training... >> training.log
cargo run --release --features cuda --bin akasha-core >> training.log 2>&1
echo [%date% %time%] Process exited -- restarting in 10 seconds... >> training.log
rem `timeout` requires an interactive console and fails instantly (no actual
rem delay) when this script runs detached/hidden with redirected output --
rem observed as hundreds of restart-loop iterations per second instead of a
rem real 10s pause. `ping` to localhost with a packet count is the standard
rem batch-script delay that works in any context.
ping -n 11 127.0.0.1 >nul
goto loop
