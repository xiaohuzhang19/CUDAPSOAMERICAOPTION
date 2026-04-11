#define n_Dim %d

/* update particle positions & velocities; each thread handles one particle */
__global__ void searchGrid(
    float *position,                /* [nDim, nFish] */
    float *velocity,                /* [nDim, nFish] */
    const float *pbest_pos,         /* [nDim, nFish] */
    const float *gbest_pos,         /* [nDim] */
    const float *r1,                /* [nDim, nFish] */
    const float *r2,                /* [nDim, nFish] */
    const float w,
    const float c1,
    const float c2,
    const int nFish                 /* passed explicitly (replaces get_global_size) */
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nFish) return;

    for (int i = 0; i < n_Dim; i++) {
        int idx = i * nFish + gid;

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
}
