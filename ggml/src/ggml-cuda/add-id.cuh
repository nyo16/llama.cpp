#include "common.cuh"

void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// glu = GLU(gate_biased, up_biased) where both inputs are ADD_ID(mul_mat_id output, bias, ids): computes
// glu(gate + gate_b[ids], up + up_b[ids]) in one pass, without writing the biased projections
bool ggml_cuda_add_id_glu_supported(const ggml_tensor * glu);
void ggml_cuda_op_add_id_glu(ggml_backend_cuda_context & ctx, ggml_tensor * glu);
