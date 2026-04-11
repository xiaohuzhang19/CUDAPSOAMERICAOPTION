#define n_PATH %d
#define n_PERIOD %d

#ifndef STRIDE
#define STRIDE 3
#endif


__device__ void mat3x3_inverse_optimized(const float XTX[9], float XTX_inv[9]) {
    float det = XTX[0]*(XTX[4]*XTX[8] - XTX[5]*XTX[7])
              - XTX[1]*(XTX[3]*XTX[8] - XTX[5]*XTX[6])
              + XTX[2]*(XTX[3]*XTX[7] - XTX[4]*XTX[6]);

    float inv_det = 1.0f / (det + 1e-15f);  /* avoid division by zero */

    XTX_inv[0] = (XTX[4]*XTX[8] - XTX[7]*XTX[5]) * inv_det;
    XTX_inv[1] = (XTX[2]*XTX[7] - XTX[1]*XTX[8]) * inv_det;
    XTX_inv[2] = (XTX[1]*XTX[5] - XTX[2]*XTX[4]) * inv_det;
    XTX_inv[3] = (XTX[5]*XTX[6] - XTX[3]*XTX[8]) * inv_det;
    XTX_inv[4] = (XTX[0]*XTX[8] - XTX[2]*XTX[6]) * inv_det;
    XTX_inv[5] = (XTX[2]*XTX[3] - XTX[0]*XTX[5]) * inv_det;
    XTX_inv[6] = (XTX[3]*XTX[7] - XTX[6]*XTX[4]) * inv_det;
    XTX_inv[7] = (XTX[1]*XTX[6] - XTX[0]*XTX[7]) * inv_det;
    XTX_inv[8] = (XTX[0]*XTX[4] - XTX[1]*XTX[3]) * inv_det;
}


__global__ void preCalcAll_Optimized(
    const float* St,
    const float K,
    const int opt,
    float* X_big_T,
    float* Xdagger_big)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_PERIOD) return;

    const int X_offset = gid * STRIDE * n_PATH;
    int itm = 0;

    /* Phase 1: collect ITM paths branchlessly */
    for (int i = 0; i < n_PATH; i++) {
        float payoff = fmaxf(0.0f, (K - St[gid + i*n_PERIOD]) * opt);
        int is_itm = (payoff > 1e-6f);

        X_big_T[X_offset + itm + 0*n_PATH] = is_itm * 1.0f;
        X_big_T[X_offset + itm + 1*n_PATH] = is_itm * payoff;
        X_big_T[X_offset + itm + 2*n_PATH] = is_itm * payoff * payoff;

        itm += is_itm;
    }

    /* Phase 2: compute XTX with fused-multiply-add */
    float XTX[9];
    for (int i=0; i<9; i++) XTX[i] = 0.0f;

    for (int i = 0; i < itm; i++) {
        float X0 = X_big_T[X_offset + i + 0*n_PATH];
        float X1 = X_big_T[X_offset + i + 1*n_PATH];
        float X2 = X_big_T[X_offset + i + 2*n_PATH];

        XTX[0] = fmaf(X0, X0, XTX[0]);
        XTX[1] = fmaf(X0, X1, XTX[1]);
        XTX[2] = fmaf(X0, X2, XTX[2]);
        XTX[4] = fmaf(X1, X1, XTX[4]);
        XTX[5] = fmaf(X1, X2, XTX[5]);
        XTX[8] = fmaf(X2, X2, XTX[8]);
    }
    XTX[3] = XTX[1];
    XTX[6] = XTX[2];
    XTX[7] = XTX[5];

    /* Phase 3: matrix inversion */
    float XTX_inv[9];
    mat3x3_inverse_optimized(XTX, XTX_inv);

    /* Phase 4: compute Xdagger with fused-multiply-add */
    for (int i = 0; i < itm; i++) {
        float X[3] = {
            X_big_T[X_offset + i + 0*n_PATH],
            X_big_T[X_offset + i + 1*n_PATH],
            X_big_T[X_offset + i + 2*n_PATH]
        };

        Xdagger_big[X_offset + i + 0*n_PATH] =
            fmaf(XTX_inv[0], X[0],
            fmaf(XTX_inv[1], X[1],
            XTX_inv[2] * X[2]));

        Xdagger_big[X_offset + i + 1*n_PATH] =
            fmaf(XTX_inv[3], X[0],
            fmaf(XTX_inv[4], X[1],
            XTX_inv[5] * X[2]));

        Xdagger_big[X_offset + i + 2*n_PATH] =
            fmaf(XTX_inv[6], X[0],
            fmaf(XTX_inv[7], X[1],
            XTX_inv[8] * X[2]));
    }
}
