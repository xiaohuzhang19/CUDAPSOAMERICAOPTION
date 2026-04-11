#define n_PATH %d
#define n_PERIOD %d

/*
 * psoAmerOption_gb  - original backward scan storing boundary/exercise arrays
 * psoAmerOption_gb2 - forward scan with break (faster for early-exercise paths)
 * psoAmerOption_gb3 - backward scan branchless (select-style, WINNER)
 *
 * Each thread handles one particle/fish.
 * nParticle is passed explicitly (replaces get_global_size).
 */


__global__ void psoAmerOption_gb(
    const float *St,
    const float *pso,           /* [nDim, nFish] updated positions */
    float *C_hat,
    int *boundary_idx,          /* [nPath, nFish] scratch */
    float *exercise,            /* [nPath, nFish] scratch */
    const float r,
    const float T,
    const float K,
    const int opt,
    const int nParticle
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nParticle) return;

    int boundary_gid;
    int St_T_idx;

    /* init scratch buffers */
    for (int path = 0; path < n_PATH; path++){
        boundary_gid = gid + path * nParticle;
        St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;
        boundary_idx[boundary_gid] = n_PERIOD - 1;
        exercise[boundary_gid] = St[St_T_idx];
    }

    /* backward scan: find first crossing per path */
    for (int prd = n_PERIOD - 1; prd > -1; prd--){
        float cur_fish_val = pso[gid + prd * nParticle];

        for (int path = 0; path < n_PATH; path++){
            float cur_St_val = St[prd + path * n_PERIOD];
            boundary_gid = gid + path * nParticle;

            if (cur_fish_val >= cur_St_val){
                boundary_idx[boundary_gid] = prd;
                exercise[boundary_gid] = cur_St_val;
            }
        }
    }

    /* compute C_hat */
    float tmp_C = 0.0f;
    float dt = T / n_PERIOD;

    for (int path = 0; path < n_PATH; path++){
        boundary_gid = gid + path * nParticle;
        tmp_C += expf(-r * (boundary_idx[boundary_gid]+1) * dt)
               * fmaxf(0.0f, (K - exercise[boundary_gid]) * opt);
    }

    C_hat[gid] = tmp_C / n_PATH;
}


/* forward scan with early break */
__global__ void psoAmerOption_gb2(
    const float *St,
    const float *pso,
    float *C_hat,
    const float r,
    const float T,
    const float K,
    const int opt,
    const int nParticle
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nParticle) return;

    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f;

    for (int path = 0; path < n_PATH; path++){
        int bound_idx = n_PERIOD - 1;
        float early_excise = St[(n_PERIOD - 1) + path * n_PERIOD];

        for (int prd = 0; prd < n_PERIOD; prd++){
            float cur_fish_val = pso[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            if (cur_fish_val >= cur_St_val){
                bound_idx = prd;
                early_excise = cur_St_val;
                break;
            }
        }

        tmp_cost += expf(-r * (bound_idx+1) * dt) * fmaxf(0.0f, (K - early_excise)*opt);
    }

    C_hat[gid] = tmp_cost / n_PATH;
}


/* backward scan branchless (replaces OpenCL select/isgreaterequal) */
__global__ void psoAmerOption_gb3(
    const float *St,
    const float *pso,
    float *C_hat,
    const float r,
    const float T,
    const float K,
    const int opt,
    const int nParticle
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nParticle) return;

    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f;

    for (int path = 0; path < n_PATH; path++){
        int bound_idx = n_PERIOD - 1;
        float early_excise = St[(n_PERIOD - 1) + path * n_PERIOD];

        for (int prd = n_PERIOD-1; prd > -1; prd--){
            float cur_fish_val = pso[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            /* branchless select: replaces OpenCL select(a, b, cond) */
            int cross = (cur_fish_val >= cur_St_val);
            bound_idx    = cross ? prd          : bound_idx;
            early_excise = cross ? cur_St_val   : early_excise;
        }

        tmp_cost += expf(-r * (bound_idx+1) * dt) * fmaxf(0.0f, (K - early_excise)*opt);
    }

    C_hat[gid] = tmp_cost / n_PATH;
}
