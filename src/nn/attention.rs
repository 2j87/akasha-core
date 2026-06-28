use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScaleMeta {
    seq_len: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HeadMoveMeta {
    seq_len: u32,
    full_dim: u32,
    head_dim: u32,
    head_offset: u32,
}

/// Multi-head self-attention. Q/K/V/grad_q/grad_k/grad_v buffers are
/// [seq_len, dim]-shaped (heads occupy disjoint, contiguous column ranges of
/// width `head_dim` within each row). Each head's compute reuses the existing
/// MatMul/MatMulTrp/MatMulWeightBwd/CausalMask/Softmax/SoftmaxBwd kernels on a
/// gathered [seq_len, head_dim] contiguous scratch buffer, bracketed by
/// HeadGather/HeadScatter to move data in/out of the shared wide buffers.
/// Head outputs are written into disjoint columns of `out_buffer`, which is
/// exactly the concatenation step -- no separate concat kernel is needed.
///
/// Meta field order for MatMul/MatMulTrp is the literal [M, N, K] tuple
/// (verified empirically against a CPU reference -- see
/// wilupgu/examples/matmul_meta_check*.rs); this differs from what the
/// pre-existing single-head implementation passed, which mismatched N/K.
pub struct SelfAttention {
    pub out_buffer: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl SelfAttention {
    pub fn new(
        ctx: Arc<WgpuContext>,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        q_buf: &Arc<Tensor>,
        k_buf: &Arc<Tensor>,
        v_buf: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_q: &Arc<Tensor>,
        grad_k: &Arc<Tensor>,
        grad_v: &Arc<Tensor>,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out_size = (seq_len * dim) as usize;
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; out_size]));

        let head_size = (seq_len * head_dim) as usize;
        let scores_size = (seq_len * seq_len) as usize;

        let shader_gather = BuiltInShader::HeadGather.get_def();
        let shader_scatter = BuiltInShader::HeadScatter.get_def();
        let shader_zero = BuiltInShader::ZeroTensor.get_def();
        let shader_qkt = BuiltInShader::MatMulTrp.get_def();
        let shader_mask = BuiltInShader::CausalMask.get_def();
        let shader_softmax = BuiltInShader::Softmax.get_def();
        let shader_out = BuiltInShader::MatMul.get_def();
        let shader_softmax_bwd = BuiltInShader::SoftmaxBwd.get_def();
        let shader_matmul = BuiltInShader::MatMul.get_def();
        let shader_matmul_trp = BuiltInShader::MatMulTrp.get_def();
        let shader_weight_bwd = BuiltInShader::MatMulWeightBwd.get_def();

        let t_meta_seq = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[AttnScaleMeta { seq_len, scale }],
        ));

        let grid_seq16 = (seq_len + 15) / 16;
        let grid_hd16 = (head_dim + 15) / 16;
        let grid_softmax = (seq_len + 255) / 256;
        let grid_zero_head = ((seq_len * head_dim) + 255) / 256;

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        // Per-head forward scores buffers must stay alive until backward
        // consumes them (softmax_bwd needs the forward Y), so they're kept
        // here rather than as locals scoped to a single loop iteration.
        let mut t_scores_heads: Vec<Arc<Tensor>> = Vec::with_capacity(num_heads as usize);
        let mut q_heads: Vec<Arc<Tensor>> = Vec::with_capacity(num_heads as usize);
        let mut k_heads: Vec<Arc<Tensor>> = Vec::with_capacity(num_heads as usize);
        let mut v_heads: Vec<Arc<Tensor>> = Vec::with_capacity(num_heads as usize);

        // Pure scratch space: each head's forward/backward fully consumes and
        // scatters these out before the next head's iteration begins, so a
        // single shared buffer (reused/overwritten sequentially per head) is
        // correct and saves num_heads-1 copies of each. Unlike t_scores/q/k/v
        // above, nothing reads these back across the forward/backward boundary
        // for a *different* head, so no cross-head lifetime is needed.
        let out_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
        let grad_out_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
        let grad_q_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
        let grad_k_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
        let grad_v_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
        let grad_y = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; scores_size]));
        let grad_raw = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; scores_size]));
        let zero_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[(seq_len * head_dim)]));

        for h in 0..num_heads {
            let head_offset = h * head_dim;
            let t_meta_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[HeadMoveMeta {
                    seq_len,
                    full_dim: dim,
                    head_dim,
                    head_offset,
                }],
            ));

            let q_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
            let k_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
            let v_head = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; head_size]));
            let t_scores = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; scores_size]));

            // ---- gather Q/K/V columns for this head ----
            for (src, dst) in [(q_buf, &q_head), (k_buf, &k_head), (v_buf, &v_head)] {
                forward_graph.add_node(
                    &shader_gather,
                    &[
                        TensorBind { binding: 0, tensor: src, mode: TensorMode::Input },
                        TensorBind { binding: 1, tensor: dst, mode: TensorMode::Output },
                        TensorBind { binding: 2, tensor: &t_meta_head, mode: TensorMode::Meta },
                    ],
                    [grid_hd16, grid_seq16, 1],
                );
            }

            // ---- scores = Q_h @ K_h^T, meta {M=seq,N=seq,K=head_dim} ----
            let meta_qkt = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, seq_len, head_dim]));
            forward_graph.add_node(
                &shader_qkt,
                &[
                    TensorBind { binding: 0, tensor: &q_head, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &k_head, mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &t_scores, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_qkt, mode: TensorMode::Meta },
                ],
                [grid_seq16, grid_seq16, 1],
            );

            forward_graph.add_node(
                &shader_mask,
                &[
                    TensorBind { binding: 0, tensor: &t_scores, mode: TensorMode::InOut },
                    TensorBind { binding: 1, tensor: &t_meta_seq, mode: TensorMode::Meta },
                ],
                [grid_seq16, grid_seq16, 1],
            );

            forward_graph.add_node(
                &shader_softmax,
                &[
                    TensorBind { binding: 0, tensor: &t_scores, mode: TensorMode::InOut },
                    TensorBind { binding: 1, tensor: &t_meta_seq, mode: TensorMode::Meta },
                ],
                [grid_softmax, 1, 1],
            );

            // ---- out_h = scores @ V_h, meta {M=seq,N=head_dim,K=seq} ----
            let meta_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, head_dim, seq_len]));
            forward_graph.add_node(
                &shader_out,
                &[
                    TensorBind { binding: 0, tensor: &t_scores, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &v_head, mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &out_head, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_out, mode: TensorMode::Meta },
                ],
                [grid_hd16, grid_seq16, 1],
            );

            // ---- scatter this head's output into out_buffer's column slice ----
            forward_graph.add_node(
                &shader_scatter,
                &[
                    TensorBind { binding: 0, tensor: &out_head, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &out_buffer, mode: TensorMode::Output },
                    TensorBind { binding: 2, tensor: &t_meta_head, mode: TensorMode::Meta },
                ],
                [grid_hd16, grid_seq16, 1],
            );

            t_scores_heads.push(t_scores);
            q_heads.push(q_head);
            k_heads.push(k_head);
            v_heads.push(v_head);

            // ============================== BACKWARD (this head) ==============================
            // gather this head's slice of the upstream output gradient
            backward_graph.add_node(
                &shader_gather,
                &[
                    TensorBind { binding: 0, tensor: grad_output, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &grad_out_head, mode: TensorMode::Output },
                    TensorBind { binding: 2, tensor: &t_meta_head, mode: TensorMode::Meta },
                ],
                [grid_hd16, grid_seq16, 1],
            );

            // dV_h = Y^T @ dOut_h  (accumulating MatMulWeightBwd -- zero first)
            backward_graph.add_node(
                &shader_zero,
                &[
                    TensorBind { binding: 0, tensor: &grad_v_head, mode: TensorMode::Output },
                    TensorBind { binding: 1, tensor: &zero_meta, mode: TensorMode::Meta },
                ],
                [grid_zero_head, 1, 1],
            );
            let meta_dv = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, head_dim, seq_len]));
            backward_graph.add_node(
                &shader_weight_bwd,
                &[
                    TensorBind { binding: 0, tensor: &t_scores_heads[h as usize], mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &grad_out_head, mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &grad_v_head, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_dv, mode: TensorMode::Meta },
                ],
                [(head_dim + 15) / 16, grid_seq16, 1],
            );

            // dY_h = dOut_h @ V_h^T, meta {M=seq,N=seq,K=head_dim}
            let meta_dy = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, seq_len, head_dim]));
            backward_graph.add_node(
                &shader_matmul_trp,
                &[
                    TensorBind { binding: 0, tensor: &grad_out_head, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &v_heads[h as usize], mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &grad_y, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_dy, mode: TensorMode::Meta },
                ],
                [grid_seq16, grid_seq16, 1],
            );

            // dRaw_h = softmax_bwd(Y_h, dY_h) -- scale folded in, as in the original design
            backward_graph.add_node(
                &shader_softmax_bwd,
                &[
                    TensorBind { binding: 0, tensor: &t_scores_heads[h as usize], mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &grad_y, mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &grad_raw, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &t_meta_seq, mode: TensorMode::Meta },
                ],
                [grid_softmax, 1, 1],
            );

            // dQ_h = dRaw_h @ K_h, meta {M=seq,N=head_dim,K=seq} (plain MatMul, no accumulation)
            let meta_dq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, head_dim, seq_len]));
            backward_graph.add_node(
                &shader_matmul,
                &[
                    TensorBind { binding: 0, tensor: &grad_raw, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &k_heads[h as usize], mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &grad_q_head, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_dq, mode: TensorMode::Meta },
                ],
                [grid_hd16, grid_seq16, 1],
            );

            // dK_h = dRaw_h^T @ Q_h (accumulating MatMulWeightBwd -- zero first)
            backward_graph.add_node(
                &shader_zero,
                &[
                    TensorBind { binding: 0, tensor: &grad_k_head, mode: TensorMode::Output },
                    TensorBind { binding: 1, tensor: &zero_meta, mode: TensorMode::Meta },
                ],
                [grid_zero_head, 1, 1],
            );
            let meta_dk = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[seq_len, head_dim, seq_len]));
            backward_graph.add_node(
                &shader_weight_bwd,
                &[
                    TensorBind { binding: 0, tensor: &grad_raw, mode: TensorMode::Input },
                    TensorBind { binding: 1, tensor: &q_heads[h as usize], mode: TensorMode::Input },
                    TensorBind { binding: 2, tensor: &grad_k_head, mode: TensorMode::Output },
                    TensorBind { binding: 3, tensor: &meta_dk, mode: TensorMode::Meta },
                ],
                [(head_dim + 15) / 16, grid_seq16, 1],
            );

            // scatter dQ_h/dK_h/dV_h into the shared [seq_len, dim] gradient buffers
            for (src, dst) in [(&grad_q_head, grad_q), (&grad_k_head, grad_k), (&grad_v_head, grad_v)] {
                backward_graph.add_node(
                    &shader_scatter,
                    &[
                        TensorBind { binding: 0, tensor: src, mode: TensorMode::Input },
                        TensorBind { binding: 1, tensor: dst, mode: TensorMode::Output },
                        TensorBind { binding: 2, tensor: &t_meta_head, mode: TensorMode::Meta },
                    ],
                    [grid_hd16, grid_seq16, 1],
                );
            }
        }

        Self {
            out_buffer,
            forward_graph,
            backward_graph,
        }
    }
}

impl Layer for SelfAttention {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}
