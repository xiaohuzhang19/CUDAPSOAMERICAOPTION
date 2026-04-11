#define n_Dim %d
#define n_Fish %d

/* each thread handles one fish */
__global__ void update_pbest(
    float *costs,
    float *pbest_costs,
    float *position,
    float *pbest_pos,
    const int nParticle             /* passed explicitly */
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nParticle) return;

    if (costs[gid] > pbest_costs[gid]) {
        pbest_costs[gid] = costs[gid];

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
    /* launched with nDim threads */
    int pbest_pos_id = gbest_id + gid * n_Fish;
    gbest_pos[gid] = pbest_pos[pbest_pos_id];
}
