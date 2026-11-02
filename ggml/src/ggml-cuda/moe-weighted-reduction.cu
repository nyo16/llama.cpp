#include "moe-weighted-reduction.cuh"

template <bool has_bias>
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  const float * __restrict__ bias,
                                                  const int32_t * __restrict__ ids,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const size_t  nb_bias1,
                                                  const size_t  nb_ids1) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    // expert output, plus its bias the way ADD_ID would have rounded it in
    const auto expert_value = [&](const int expert, const uint64_t row) {
        float value = experts[row * n_embd + col];
        if constexpr (has_bias) {
            const int32_t e = *(const int32_t *) ((const char *) ids + token * nb_ids1 + expert * sizeof(int32_t));
            value += ((const float *) ((const char *) bias + e * nb_bias1))[col];
        }
        return value;
    };

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (expert_value(0, first_row) * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (expert_value(expert, row) * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          const float * bias,
                                          const int32_t * ids,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          size_t        nb_bias1,
                                          size_t        nb_ids1,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    if (bias != nullptr) {
        moe_weighted_reduction_f32<true><<<blocks, threads, 0, stream>>>(
            experts, expert_scale, weights, bias, ids, dst, n_embd, n_expert_used, nb_bias1, nb_ids1);
    } else {
        moe_weighted_reduction_f32<false><<<blocks, threads, 0, stream>>>(
            experts, expert_scale, weights, nullptr, nullptr, dst, n_embd, n_expert_used, 0, 0);
    }
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         bias) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const ggml_tensor * bias_b   = bias ? bias->src[1] : nullptr;
    const ggml_tensor * bias_ids = bias ? bias->src[2] : nullptr;
    if (bias) {
        GGML_ASSERT(bias->op == GGML_OP_ADD_ID && bias->src[0] == experts);
        GGML_ASSERT(bias_b->type == GGML_TYPE_F32 && bias_b->nb[0] == sizeof(float) && bias_b->ne[0] == experts->ne[0]);
        GGML_ASSERT(bias_ids->type == GGML_TYPE_I32 && bias_ids->nb[0] == sizeof(int32_t));
        GGML_ASSERT(experts->ne[2] * experts->ne[3] == bias_ids->ne[1] && experts->ne[1] == bias_ids->ne[0]);
    }

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  bias_b ? (const float *) bias_b->data : nullptr,
                                  bias_ids ? (const int32_t *) bias_ids->data : nullptr,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used,
                                  bias_b ? bias_b->nb[1] : 0, bias_ids ? bias_ids->nb[1] : 0, stream);
    CUDA_CHECK(cudaGetLastError());
}
