#include "add-id.cuh"
#include "unary.cuh"

static __global__ void add_id_kernel(
        const float * src0, const float * src1, const int32_t * src2, float * dst,
        int64_t ne0, int64_t ne1,
        size_t nb01, size_t nb02,
        size_t nb11,
        size_t nb21
    ) {

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;

    const int i11 = *(const int32_t *) ((const char *) src2 + i1*sizeof(int32_t) + i2*nb21);

    const size_t nb1 = ne0 * sizeof(float);
    const size_t nb2 = ne1 * nb1;

    float * dst_row = (float *)((char *)dst + i1*nb1 + i2*nb2);
    const float * src0_row = (const float *)((const char *)src0 +  i1*nb01 + i2*nb02);
    const float * src1_row = (const float *)((const char *)src1 + i11*nb11);

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const int32_t * src2_d = (const int32_t *)src2->data;
    float * dst_d = (float *)dst->data;

    int threads = std::min((int)ne00, 768); // cols
    dim3 blocks(ne01, ne02); // n_experts_used, n_tokens
    add_id_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        src0_d, src1_d, src2_d, dst_d,
        ne0, ne1,
        nb01, nb02,
        nb11,
        nb21
    );
}

template <ggml_glu_op glu_op>
static __global__ void add_id_glu_kernel(
        const float * __restrict__ gate, const float * __restrict__ up,
        const float * __restrict__ gate_b, const float * __restrict__ up_b, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const int64_t ne0, const int64_t ne1, const int64_t n,
        const size_t nb_gate_b1, const size_t nb_up_b1, const size_t nb_ids1, const float alpha, const float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t i0  = i % ne0;
    const int64_t i12 = i / ne0;
    const int64_t i1  = i12 % ne1;
    const int64_t i2  = i12 / ne1;

    const int32_t e = *(const int32_t *) ((const char *) ids + i2*nb_ids1 + i1*sizeof(int32_t));

    // the same two roundings as ADD_ID followed by the GLU kernel
    const float g = gate[i] + ((const float *) ((const char *) gate_b + e*nb_gate_b1))[i0];
    const float u = up[i]   + ((const float *) ((const char *) up_b   + e*nb_up_b1))[i0];

    if constexpr (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
        dst[i] = ggml_cuda_op_swiglu_oai_single(g, u, alpha, limit);
    } else {
        static_assert(glu_op == GGML_GLU_OP_SWIGLU, "unsupported GLU");
        dst[i] = ggml_cuda_op_silu_single(g) * u;
    }
}

bool ggml_cuda_add_id_glu_supported(const ggml_tensor * glu) {
    if (glu->op != GGML_OP_GLU || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return false;
    }
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_SWIGLU_OAI) {
        return false;
    }
    for (const ggml_tensor * biased : { glu->src[0], glu->src[1] }) {
        if (biased == nullptr || biased->op != GGML_OP_ADD_ID || !ggml_are_same_shape(biased, glu)) {
            return false;
        }
        const ggml_tensor * x    = biased->src[0];
        const ggml_tensor * bias = biased->src[1];
        const ggml_tensor * ids  = biased->src[2];
        if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || bias->type != GGML_TYPE_F32 || bias->nb[0] != sizeof(float) ||
                ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t)) {
            return false;
        }
    }
    // both lanes route through the same experts
    return glu->src[0]->src[2] == glu->src[1]->src[2];
}

void ggml_cuda_op_add_id_glu(ggml_backend_cuda_context & ctx, ggml_tensor * glu) {
    GGML_ASSERT(ggml_cuda_add_id_glu_supported(glu));

    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    const ggml_tensor * ids  = gate->src[2];

    const int64_t n     = ggml_nelements(glu);
    const float   alpha = ggml_get_op_params_f32(glu, 2);
    const float   limit = ggml_get_op_params_f32(glu, 3);

    const int64_t num_blocks = (n + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const dim3 grid(num_blocks, 1, 1);
    const dim3 block(CUDA_GLU_BLOCK_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

#define ADD_ID_GLU_ARGS \
    (const float *) gate->src[0]->data, (const float *) up->src[0]->data, \
    (const float *) gate->src[1]->data, (const float *) up->src[1]->data, (const int32_t *) ids->data, \
    (float *) glu->data, glu->ne[0], glu->ne[1], n, gate->src[1]->nb[1], up->src[1]->nb[1], ids->nb[1], alpha, limit

    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI) {
        add_id_glu_kernel<GGML_GLU_OP_SWIGLU_OAI><<<grid, block, 0, stream>>>(ADD_ID_GLU_ARGS);
    } else {
        add_id_glu_kernel<GGML_GLU_OP_SWIGLU><<<grid, block, 0, stream>>>(ADD_ID_GLU_ARGS);
    }
#undef ADD_ID_GLU_ARGS
}
