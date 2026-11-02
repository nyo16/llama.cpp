#include "common.cuh"

// bias: an ADD_ID node in front of the reduction; its src[0] is then passed as experts, and its expert
// biases (src[1], selected by src[2]) are added before the scale and router weight
void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         bias = nullptr);
