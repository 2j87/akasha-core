use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct Linear {
    pub weight: Arc<Tensor>,
    pub out_buffer: Arc<Tensor>,
    pub grad_weight: Arc<Tensor>,
    pub grad_input: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl Linear {
    pub fn new(
        ctx: Arc<WgpuContext>,
        in_features: u32,
        out_features: u32,
        seq_len: u32,
        weight_data: &[Real],
        input_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_input: &Arc<Tensor>,
    ) -> Self {
        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), weight_data));

        let m = seq_len;
        let meta_data = vec![m, out_features, in_features];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let out_size = (m * out_features) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let zero_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_w));

        let grad_input = grad_input.clone();

        // --- FORWARD ---
        let shader_fw = BuiltInShader::MatMul.get_def();
        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            &shader_fw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &weight,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &out_buffer,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(out_features + 15) / 16, (m + 15) / 16, 1],
        );

        // --- BACKWARD ---
        let shader_bwd_w = BuiltInShader::MatMulWeightBwd.get_def();
        let shader_bwd_in = BuiltInShader::MatMulTrp.get_def(); // Input Trp B
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        // grad_input = grad_output[M,out_features] @ weight^T. MatMulTrp's
        // convention is C[M,N] = A[M,K] @ B^T where B is stored as [N,K].
        // `weight` is physically stored [in_features,out_features] (its
        // forward-pass [K,N] role), which is exactly [N,K] for *this* call
        // with N=in_features, K=out_features -- the opposite labeling from
        // the forward/grad_weight meta (which has N=out_features,
        // K=in_features). Reusing `t_meta` here swaps N/K and silently
        // computes the wrong matrix (confirmed via a CUDA-vs-Vulkan A/B: both
        // backends "faithfully" mis-execute the swapped meta, but via
        // different mechanics -- cuBLAS's column-major reinterpretation vs
        // WGSL's direct strided indexing -- producing different wrong
        // results instead of merely matching wrong results).
        let meta_grad_in_data = vec![m, in_features, out_features];
        let t_meta_grad_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_grad_in_data));

        backward_graph.add_node(
            &shader_bwd_w,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_weight,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(out_features + 15) / 16, (in_features + 15) / 16, 1],
        );

        backward_graph.add_node(
            &shader_bwd_in,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &weight,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_input,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_grad_in,
                    mode: TensorMode::Meta,
                },
            ],
            [(in_features + 15) / 16, (m + 15) / 16, 1],
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl Layer for Linear {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}
