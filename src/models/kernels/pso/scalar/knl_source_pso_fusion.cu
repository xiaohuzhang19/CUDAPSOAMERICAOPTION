#define n_Dim %d
#define n_PATH %d
#define n_PERIOD %d
#define n_Fish %d

/*
 * Fused kernel: searchGrid + fitness (American option) + pbest update.
 * Each thread handles one particle/fish.
 * nParticle is passed explicitly.
 */
__global__ void pso(
    /* searchGrid */
    float *position,                /* [nDim, nFish] */
    float *velocity,                /* [nDim, nFish] */
    float *pbest_pos,               /* [nDim, nFish] */
    const float *gbest_pos,         /* [nDim] */
    const float *r1,                /* [nDim, nFish] */
    const float *r2,                /* [nDim, nFish] */
    const float w,
    const float c1,
    const float c2,
    /* American option fitness */
    const float *St,
    const float r,
    const float T,
    const float K,
    const int opt,
    /* update pbest */
    float *pbest_costs,
    /* explicit size */
    const int nParticle
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nParticle) return;

    /* 1. searchGrid */
    for (int i = 0; i < n_Dim; i++) {
        int idx = i * nParticle + gid;

        float pos   = position[idx];
        float vel   = velocity[idx];
        float pbest = pbest_pos[idx];
        float r1_v  = r1[idx];
        float r2_v  = r2[idx];
        float gbest = gbest_pos[i];

        vel = w * vel + c1 * r1_v * (pbest - pos) + c2 * r2_v * (gbest - pos);
        pos += vel;

        velocity[idx] = vel;
        position[idx] = pos;
    }

    /* 2. fitness: American option backward scan */
    float dt = T / n_PERIOD;
    float tmp_C = 0.0f;

    for (int path = 0; path < n_PATH; path++){
        int bound_idx = n_PERIOD - 1;
        float early_excise = St[(n_PERIOD - 1) + path * n_PERIOD];

        for (int prd = n_PERIOD-1; prd > -1; prd--){
            float cur_fish_val = position[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            int cross = (cur_fish_val >= cur_St_val);
            bound_idx    = cross ? prd        : bound_idx;
            early_excise = cross ? cur_St_val : early_excise;
        }

        tmp_C += expf(-r * (bound_idx+1) * dt) * fmaxf(0.0f, (K - early_excise)*opt);
    }

    tmp_C /= n_PATH;

    /* 3. update pbest */
    if (tmp_C > pbest_costs[gid]) {
        pbest_costs[gid] = tmp_C;

        for (int i = 0; i < n_Dim; i++) {
            int idx = gid + i * nParticle;
            pbest_pos[idx] = position[idx];
        }
    }
}


/* each thread handles one dimension */
__global__ void update_gbest_pos(
    float *gbest_pos,
    float *pbest_pos,
    const int gbest_id
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pbest_pos_id = gbest_id + gid * n_Fish;
    gbest_pos[gid] = pbest_pos[pbest_pos_id];
}
