import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ==========================================
# 1. THE RAW CUDA C++ KERNEL
# ==========================================
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scorer_kernel(
    const float* z, const float* belief,
    const float* log_var, float beta_a, float bias_a,
    const float* mu_clean, float temp_b, float bias_b,
    float* r_a_out, float* r_b_out, float* md_clean_out,
    int batch_size, int latent_dim
) {
    // 1. Thread mapping (1 thread per batch item/frame)
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float md_a = 0.0f;
    float md_b = 0.0f;

    const float* z_row = z + b * latent_dim;
    const float* belief_row = belief + b * latent_dim;

    // 2. Compute distances across the latent dimension
    for (int d = 0; d < latent_dim; ++d) {
        // Brain A math
        float diff_a = z_row[d] - belief_row[d];
        float lv = log_var[d];
        lv = fminf(fmaxf(lv, -10.0f), 10.0f); // Clamp
        float inv_var = expf(-lv);
        md_a += diff_a * diff_a * inv_var;

        // Brain B math (Simplified Euclidean for baseline)
        float diff_b = z_row[d] - mu_clean[d];
        md_b += diff_b * diff_b;
    }

    // 3. Finalize Brain A Sigmoid
    float raw_a = -0.5f * md_a;
    float rel_a_raw = beta_a * raw_a + bias_a;
    r_a_out[b] = 1.0f / (1.0f + expf(-rel_a_raw));

    // 4. Finalize Brain B Sigmoid
    float md_clean_val = sqrtf(md_b);
    float rel_b_raw = (-md_clean_val / temp_b) + bias_b;
    r_b_out[b] = 1.0f / (1.0f + expf(-rel_b_raw));
    md_clean_out[b] = md_clean_val;
}

std::vector<torch::Tensor> fused_scorer_forward(
    torch::Tensor z, torch::Tensor belief,
    torch::Tensor log_var, float beta_a, float bias_a,
    torch::Tensor mu_clean, float temp_b, float bias_b
) {
    int batch_size = z.size(0);
    int latent_dim = z.size(1);

    auto r_a_out = torch::empty({batch_size}, z.options());
    auto r_b_out = torch::empty({batch_size}, z.options());
    auto md_clean_out = torch::empty({batch_size}, z.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    fused_scorer_kernel<<<blocks, threads>>>(
        z.data_ptr<float>(), belief.data_ptr<float>(),
        log_var.data_ptr<float>(), beta_a, bias_a,
        mu_clean.data_ptr<float>(), temp_b, bias_b,
        r_a_out.data_ptr<float>(), r_b_out.data_ptr<float>(), md_clean_out.data_ptr<float>(),
        batch_size, latent_dim
    );

    return {r_a_out, r_b_out, md_clean_out};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_scorer_forward(
    torch::Tensor z, torch::Tensor belief,
    torch::Tensor log_var, float beta_a, float bias_a,
    torch::Tensor mu_clean, float temp_b, float bias_b
);
"""

# ==========================================
# 2. COMPILE ON THE FLY
# ==========================================
fused_module = load_inline(
    name='fused_scorer_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_scorer_forward'],
    with_cuda=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3']
)

# ==========================================
# 3. THE PYTORCH WRAPPER
# ==========================================
class FusedBrainScorer(nn.Module):
    """Wrapper to replace the separate Brain A and Brain B calls."""
    def __init__(self, brain_a, brain_b):
        super().__init__()
        self.brain_a = brain_a
        self.brain_b = brain_b
        
    def forward(self, belief, z):
        # Ensure memory is contiguous for the C++ pointers
        z_contig = z.contiguous().float()
        belief_contig = belief.contiguous().float()
        
        r_a, r_b, md_clean = fused_module.fused_scorer_forward(
            z_contig, 
            belief_contig,
            self.brain_a.log_var.contiguous().float(),
            self.brain_a.beta.item(),
            self.brain_a.bias.item(),
            self.brain_b.mu_clean.contiguous().float(),
            self.brain_b.temperature,
            self.brain_b.bias
        )
        return r_a, r_b, md_clean