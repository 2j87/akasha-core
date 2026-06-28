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
timeout /t 10
goto loop
