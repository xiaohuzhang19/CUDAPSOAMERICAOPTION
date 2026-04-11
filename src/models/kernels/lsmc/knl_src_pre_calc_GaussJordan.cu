#define n_PATH %d
#define n_PERIOD %d

#ifndef STRIDE
#define STRIDE 3
#endif


__global__ void preCalcAll_GaussJordan(float* St, float K, int opt, float* X_big_T, float* Xdagger_big){

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n_PERIOD) return;

    int X_big_T_start_idx = gid * STRIDE * n_PATH;
    int itm = 0;

    float X0 = 0.0f;
    float X1 = 0.0f;
    float X2 = 0.0f;

    for (int i=0; i<n_PATH; i++){
        float payoff = fmaxf(0.0f, (K - St[gid + i * n_PERIOD]) * opt);

        if (payoff > 0){
            X_big_T[X_big_T_start_idx + itm + 0 * n_PATH] = 1.0f;
            X_big_T[X_big_T_start_idx + itm + 1 * n_PATH] = payoff;
            X_big_T[X_big_T_start_idx + itm + 2 * n_PATH] = payoff * payoff;
            itm++;
        }
    }

    float XTX[STRIDE * STRIDE];
    for (int i=0; i<STRIDE*STRIDE; i++) XTX[i] = 0.0f;

    for (int i=0; i<itm; i++){
        X0 = X_big_T[X_big_T_start_idx + i + 0 * n_PATH];
        X1 = X_big_T[X_big_T_start_idx + i + 1 * n_PATH];
        X2 = X_big_T[X_big_T_start_idx + i + 2 * n_PATH];

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

    /* Gauss-Jordan Elimination inverse 3x3 */
    float XTX_inv[STRIDE * STRIDE];
    for (int i=0; i<STRIDE*STRIDE; i++) XTX_inv[i] = 0.0f;

    float right_joint[STRIDE][STRIDE * 2];
    for (int i=0; i<STRIDE; i++)
        for (int j=0; j<STRIDE*2; j++)
            right_joint[i][j] = 0.0f;

    float d = 0.0f;

    for (int i=0; i<STRIDE; i++)
        for (int j=0; j<STRIDE; j++)
            right_joint[i][j] = XTX[i * STRIDE + j];

    right_joint[0][3] = 1.0f;
    right_joint[1][4] = 1.0f;
    right_joint[2][5] = 1.0f;

    for (int i=STRIDE-1; i>0; i--){
        if (right_joint[i-1][1] < right_joint[i][1]){
            for (int j=0; j<STRIDE*2; j++){
                d = right_joint[i][j];
                right_joint[i][j] = right_joint[i-1][j];
                right_joint[i-1][j] = d;
            }
        }
    }

    for (int i=0; i<STRIDE; i++){
        for (int j=0; j<STRIDE; j++){
            if (j != i && right_joint[j][i] != 0){
                d = right_joint[j][i] / right_joint[i][i];
                for (int k=0; k<STRIDE*2; k++){
                    right_joint[j][k] -= right_joint[i][k] * d;
                }
            }
        }
    }

    XTX_inv[0] = right_joint[0][3] / right_joint[0][0];
    XTX_inv[1] = right_joint[0][4] / right_joint[0][0];
    XTX_inv[2] = right_joint[0][5] / right_joint[0][0];
    XTX_inv[3] = right_joint[1][3] / right_joint[1][1];
    XTX_inv[4] = right_joint[1][4] / right_joint[1][1];
    XTX_inv[5] = right_joint[1][5] / right_joint[1][1];
    XTX_inv[6] = right_joint[2][3] / right_joint[2][2];
    XTX_inv[7] = right_joint[2][4] / right_joint[2][2];
    XTX_inv[8] = right_joint[2][5] / right_joint[2][2];

    int Xdagger_big_start_idx = gid * STRIDE * n_PATH;

    for (int i=0; i<itm; i++){
        X0 = 0.0f; X1 = 0.0f; X2 = 0.0f;

        for (int k=0; k<STRIDE; k++){
            X0 += XTX_inv[0 * STRIDE + k] * X_big_T[X_big_T_start_idx + i + k * n_PATH];
            X1 += XTX_inv[1 * STRIDE + k] * X_big_T[X_big_T_start_idx + i + k * n_PATH];
            X2 += XTX_inv[2 * STRIDE + k] * X_big_T[X_big_T_start_idx + i + k * n_PATH];
        }

        Xdagger_big[Xdagger_big_start_idx + i + 0 * n_PATH] = X0;
        Xdagger_big[Xdagger_big_start_idx + i + 1 * n_PATH] = X1;
        Xdagger_big[Xdagger_big_start_idx + i + 2 * n_PATH] = X2;
    }
}
