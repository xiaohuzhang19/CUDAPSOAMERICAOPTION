#define n_PATH %d
#define n_PERIOD %d

/*
getEuroOption_optimized is the WINNER
16384 paths : getEuroOption   7.457971572875977 ms; getEuroOption_optimized 2.4869441986083984 ms
32768 paths : getEuroOption  37.24312782287598 ms;  getEuroOption_optimized 3.8788318634033203 ms
65536 paths : getEuroOption  51.766157150268555 ms; getEuroOption_optimized 3.778696060180664 ms
*/


__global__ void getEuroOption(float *Z, float S0, float K, float r, float sigma, float T,
    int opt, float *payoffs){

    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_PATH) return;

    float dt = T / n_PERIOD;
    float nudt = (r - 0.5f * sigma*sigma) * dt;
    float volsdt = sigma * sqrtf(dt);
    float lnS0 = logf(S0);

    float deltaSt = 0.0f;
    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++){
        deltaSt += nudt + volsdt * Z[cur_t + n_PERIOD * path_id];
    }
    float St = expf(deltaSt + lnS0);

    payoffs[path_id] = expf(-r*T) * fmaxf(0.0f, (K - St)*opt);
}


__global__ void getEuroOption_optimized(
    const float *Z,
    const float lnS0,
    const float K,
    const float r,
    const float sigma,
    const float T,
    const int opt,
    const float exp_neg_rT,
    float *payoffs
){
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_PATH) return;

    const float dt = T / n_PERIOD;
    const float nudt = (r - 0.5f * sigma * sigma) * dt;
    const float volsdt = sigma * sqrtf(dt);

    float deltaSt = 0.0f;
    const int z_offset = path_id * n_PERIOD;

    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++){
        deltaSt += nudt + volsdt * Z[z_offset + cur_t];
    }

    const float St = expf(deltaSt + lnS0);
    payoffs[path_id] = exp_neg_rT * fmaxf(0.0f, (K - St) * opt);
}


/*
   Only works no exceeding 2**16 paths
   Two-step parallel reduction
*/
__global__ void getEuroOption_optimized_sum1(
    const float *Z,
    const float lnS0,
    const float K,
    const float r,
    const float sigma,
    const float T,
    const int opt,
    const float exp_neg_rT,
    float *C_hats       /* partial sums output, one per block */
){
    extern __shared__ float local_sums[];

    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lid = threadIdx.x;
    const int local_size = blockDim.x;

    const float dt = T / n_PERIOD;
    const float nudt = (r - 0.5f * sigma * sigma) * dt;
    const float volsdt = sigma * sqrtf(dt);

    float val = 0.0f;
    if (path_id < n_PATH) {
        float deltaSt = 0.0f;
        const int z_offset = path_id * n_PERIOD;
        for(int cur_t = 0; cur_t < n_PERIOD; cur_t++) {
            deltaSt += nudt + volsdt * Z[z_offset + cur_t];
        }
        const float St = expf(deltaSt + lnS0);
        val = exp_neg_rT * fmaxf(0.0f, (K - St) * opt);
    }

    local_sums[lid] = val;
    __syncthreads();

    /* parallel reduction in shared memory */
    for(int stride = local_size >> 1; stride > 0; stride >>= 1){
        if(lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        __syncthreads();
    }

    if(lid == 0) {
        C_hats[blockIdx.x] = local_sums[0];
    }
}


/* Final reduction: C_hats array -> single float */
__global__ void getEuroOption_optimized_sum2(
    const int n_groups,
    float* partial_sums,
    float* final_result
){
    extern __shared__ float local_sums[];

    const int lid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + lid;
    const int local_size = blockDim.x;

    local_sums[lid] = (gid < n_groups) ? partial_sums[gid] : 0.0f;
    __syncthreads();

    for(int stride = local_size >> 1; stride > 0; stride >>= 1){
        if(lid < stride){
            local_sums[lid] += local_sums[lid + stride];
        }
        __syncthreads();
    }

    if(lid == 0){
        final_result[0] = local_sums[0];
    }
}
