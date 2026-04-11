#define n_PATH %d
#define n_PERIOD %d

#ifndef STRIDE
#define STRIDE 3
#endif


__global__ void preCalcAll_ClassicAdjoint(float* St, float K, int opt, float* X_big_T, float* Xdagger_big){

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n_PERIOD) return;

    int XbigT_start_idx = gid * STRIDE * n_PATH;
    int itm = 0;

    float X0 = 0.0f;
    float X1 = 0.0f;
    float X2 = 0.0f;

    for (int i=0; i<n_PATH; i++){
        float payoff = fmaxf(0.0f, (K - St[gid + i * n_PERIOD]) * opt);

        if (payoff > 0){
            X_big_T[XbigT_start_idx + itm + 0 * n_PATH] = 1.0f;
            X_big_T[XbigT_start_idx + itm + 1 * n_PATH] = payoff;
            X_big_T[XbigT_start_idx + itm + 2 * n_PATH] = payoff * payoff;
            itm++;
        }
    }

    float XTX[STRIDE * STRIDE];
    for (int i=0; i<STRIDE*STRIDE; i++) XTX[i] = 0.0f;

    for (int i=0; i<itm; i++){
        X0 = X_big_T[XbigT_start_idx + i + 0 * n_PATH];
        X1 = X_big_T[XbigT_start_idx + i + 1 * n_PATH];
        X2 = X_big_T[XbigT_start_idx + i + 2 * n_PATH];

        XTX[0] += X0 * X0;
        XTX[1] += X0 * X1;
        XTX[2] += X0 * X2;
        XTX[4] += X1 * X1;
        XTX[5] += X1 * X2;
        XTX[8] += X2 * X2;
    }
    XTX[3] = XTX[1];
    XTX[6] = XTX[2];
    XTX[7] = XTX[5];

    /* Classic Adjoint inverse 3x3 */
    float XTX_inv[STRIDE * STRIDE];
    for (int i=0; i<STRIDE*STRIDE; i++) XTX_inv[i] = 0.0f;

    float det = XTX[0] * (XTX[4] * XTX[8] - XTX[5] * XTX[7])
              - XTX[1] * (XTX[3] * XTX[8] - XTX[5] * XTX[6])
              + XTX[2] * (XTX[3] * XTX[7] - XTX[4] * XTX[6]);

    if (det != 0.0f) {
        float inv_det = 1.0f / det;
        XTX_inv[0] = (XTX[4] * XTX[8] - XTX[5] * XTX[7]) * inv_det;
        XTX_inv[1] = (XTX[2] * XTX[7] - XTX[1] * XTX[8]) * inv_det;
        XTX_inv[2] = (XTX[1] * XTX[5] - XTX[2] * XTX[4]) * inv_det;
        XTX_inv[3] = (XTX[5] * XTX[6] - XTX[3] * XTX[8]) * inv_det;
        XTX_inv[4] = (XTX[0] * XTX[8] - XTX[2] * XTX[6]) * inv_det;
        XTX_inv[5] = (XTX[2] * XTX[3] - XTX[0] * XTX[5]) * inv_det;
        XTX_inv[6] = (XTX[3] * XTX[7] - XTX[4] * XTX[6]) * inv_det;
        XTX_inv[7] = (XTX[1] * XTX[6] - XTX[0] * XTX[7]) * inv_det;
        XTX_inv[8] = (XTX[0] * XTX[4] - XTX[1] * XTX[3]) * inv_det;
    } else {
        /* singular: identity fallback */
        XTX_inv[0] = 1.0f; XTX_inv[4] = 1.0f; XTX_inv[8] = 1.0f;
    }

    int Xdagger_big_start_idx = gid * STRIDE * n_PATH;

    for (int i=0; i<itm; i++){
        X0 = 0.0f; X1 = 0.0f; X2 = 0.0f;

        for (int k=0; k<STRIDE; k++){
            X0 += XTX_inv[0 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];
            X1 += XTX_inv[1 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];
            X2 += XTX_inv[2 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];
        }

        Xdagger_big[Xdagger_big_start_idx + i + 0 * n_PATH] = X0;
        Xdagger_big[Xdagger_big_start_idx + i + 1 * n_PATH] = X1;
        Xdagger_big[Xdagger_big_start_idx + i + 2 * n_PATH] = X2;
    }
}
