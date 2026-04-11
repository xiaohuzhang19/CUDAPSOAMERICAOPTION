import numpy as np
import numpy.linalg as la
from pathlib import Path
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from .mc import MonteCarloBase
from .utils import CUDAEnv, alloc_and_copy, alloc_empty, copy_to_host
import time

_MODELS_DIR = Path(__file__).parent
_KERNELS_DIR = _MODELS_DIR / "kernels"


def checkError(a, b):
    return la.norm(a - b)


# Classic Adjoint (CPU helper, unchanged)
def inverse_3X3_matrix(A):
    I_Q_list = A
    det_ = I_Q_list[0][0] * (
            (I_Q_list[1][1] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][1])) - \
           I_Q_list[0][1] * (
                   (I_Q_list[1][0] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][0])) + \
           I_Q_list[0][2] * (
                   (I_Q_list[1][0] * I_Q_list[2][1]) - (I_Q_list[1][1] * I_Q_list[2][0]))
    if det_ == 0.0:
        return det_, np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)
    co_fctr_1 = [(I_Q_list[1][1]*I_Q_list[2][2])-(I_Q_list[1][2]*I_Q_list[2][1]),
                 -((I_Q_list[1][0]*I_Q_list[2][2])-(I_Q_list[1][2]*I_Q_list[2][0])),
                 (I_Q_list[1][0]*I_Q_list[2][1])-(I_Q_list[1][1]*I_Q_list[2][0])]
    co_fctr_2 = [-((I_Q_list[0][1]*I_Q_list[2][2])-(I_Q_list[0][2]*I_Q_list[2][1])),
                 (I_Q_list[0][0]*I_Q_list[2][2])-(I_Q_list[0][2]*I_Q_list[2][0]),
                 -((I_Q_list[0][0]*I_Q_list[2][1])-(I_Q_list[0][1]*I_Q_list[2][0]))]
    co_fctr_3 = [(I_Q_list[0][1]*I_Q_list[1][2])-(I_Q_list[0][2]*I_Q_list[1][1]),
                 -((I_Q_list[0][0]*I_Q_list[1][2])-(I_Q_list[0][2]*I_Q_list[1][0])),
                 (I_Q_list[0][0]*I_Q_list[1][1])-(I_Q_list[0][1]*I_Q_list[1][0])]
    inv_list = [[1/det_*co_fctr_1[0], 1/det_*co_fctr_2[0], 1/det_*co_fctr_3[0]],
                [1/det_*co_fctr_1[1], 1/det_*co_fctr_2[1], 1/det_*co_fctr_3[1]],
                [1/det_*co_fctr_1[2], 1/det_*co_fctr_2[2], 1/det_*co_fctr_3[2]]]
    return det_.astype(np.float32), np.array(inv_list).astype(np.float32)


def GJ_Elimination_inverse_3X3(A):
    B = np.zeros((3, 6), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            B[i][j] = A[i][j]
    B[0][3] = B[1][4] = B[2][5] = 1
    for i in range(2, 0, -1):
        if B[i-1][1] < B[i][1]:
            for j in range(6):
                B[i][j], B[i-1][j] = B[i-1][j], B[i][j]
    for i in range(3):
        for j in range(3):
            if j != i and B[j][i] != 0:
                d = B[j][i] / B[i][i]
                for k in range(6):
                    B[j][k] -= B[i][k] * d
    C = np.zeros((3, 3), dtype=np.float32)
    C[0][0]=B[0][3]/B[0][0]; C[0][1]=B[0][4]/B[0][0]; C[0][2]=B[0][5]/B[0][0]
    C[1][0]=B[1][3]/B[1][1]; C[1][1]=B[1][4]/B[1][1]; C[1][2]=B[1][5]/B[1][1]
    C[2][0]=B[2][3]/B[2][2]; C[2][1]=B[2][4]/B[2][2]; C[2][2]=B[2][5]/B[2][2]
    return C


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class LongStaffBase:
    def __init__(self, mc: MonteCarloBase):
        self.mc = mc


# --------------------------------------------------------------------------- #
# CPU-only LSMC (unchanged)                                                   #
# --------------------------------------------------------------------------- #

class LSMC_Numpy(LongStaffBase):
    def __init__(self, mc: MonteCarloBase, inverseType='benchmark_pinv', toggleCV='OFF', log=None):
        super().__init__(mc)
        self.inverseType = inverseType
        self.toggleCV = toggleCV
        self.log = log

    def __continuation_value(self, x, Y):
        inverseType_cpu = ["benchmark_pinv", "benchmark_lstsq", "SVD", "CA", "GJ"]
        if self.inverseType not in inverseType_cpu:
            raise Exception(f'inverseType must be one of {inverseType_cpu}')
        X = np.c_[np.ones(len(x)), x, np.square(x)].astype(np.float32)
        match self.inverseType:
            case "benchmark_pinv":
                Xdagger = np.linalg.pinv(X)
                coef_ = Xdagger @ Y
            case "benchmark_lstsq":
                coef_ = np.linalg.lstsq(X, Y, rcond=None)[0]
            case "SVD":
                U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
                Xdagger = VT.T @ np.linalg.inv(np.diag(Sigma)) @ U.T
                coef_ = Xdagger @ Y
            case "CA":
                Xdagger = inverse_3X3_matrix(X.T @ X)[1] @ X.T
                coef_ = Xdagger @ Y
            case "GJ":
                Xdagger = GJ_Elimination_inverse_3X3(X.T @ X) @ X.T
                coef_ = Xdagger @ Y
        if self.log == 'INFO':
            print('X:\n', X)
            print('Y:', Y.flatten())
            print('Xdagger:\n', Xdagger)
        cont_value = X @ coef_
        return cont_value, coef_

    def longstaff_schwartz_itm_path_fast(self):
        start = time.perf_counter()
        dt = self.mc.T / self.mc.nPeriod
        df = np.exp(-self.mc.r * dt)
        dc_cashflow_t = []
        coef_t = []
        payoffs = self.mc.getPayoffs()
        itm_allPeriod = (payoffs != 0).sum(0)
        dc_cashflow = payoffs[:, -1]

        for t in range(self.mc.nPeriod-2, -1, -1):
            if self.log == 'INFO':
                print('\ntime t:', t+1)
            dc_cashflow = dc_cashflow * df
            itm = payoffs[:, t].nonzero()
            num_itm = itm_allPeriod[t]
            x = payoffs[itm, t].reshape(num_itm, 1)
            Y = dc_cashflow[itm].reshape(num_itm, 1)
            cont_value, coef_ = self.__continuation_value(x, Y)
            coef_t.append(coef_.flatten())
            if self.log == 'INFO':
                print('coef:', coef_.flatten())
                print('pre ds cf itm:', dc_cashflow[itm])
                print('cont val     :', cont_value.flatten())
                print('exer val     :', payoffs[itm, t])
            match self.toggleCV:
                case 'ON':
                    BS_itm = self.mc.BS[itm, t].flatten()
                    max_CvBS = np.maximum(cont_value.flatten(), BS_itm)
                    dc_cashflow[itm] = np.where(payoffs[itm, t] > max_CvBS, payoffs[itm, t], dc_cashflow[itm])
                case 'OFF':
                    dc_cashflow[itm] = np.where(payoffs[itm, t] > cont_value.flatten(), payoffs[itm, t], dc_cashflow[itm])
            dc_cashflow_t.append(dc_cashflow)
            if self.log == 'INFO':
                print('post ds cf itm:', dc_cashflow)

        C_hat = np.sum(dc_cashflow) * df / self.mc.nPath
        elapse = (time.perf_counter() - start) * 1e3
        print(f'Longstaff numpy price: {C_hat} - {elapse:.3f} ms')
        return C_hat, elapse


# --------------------------------------------------------------------------- #
# GPU LSMC (PyCUDA — replaces LSMC_OpenCL)                                   #
# --------------------------------------------------------------------------- #

class LSMC_CUDA(LongStaffBase):
    """CUDA port of LSMC_OpenCL.  inverseType: 'CA' | 'GJ' | 'optimized'"""

    def __init__(self, mc: MonteCarloBase, preCalc=None, inverseType='GJ', toggleCV='OFF', log=None):
        super().__init__(mc)
        self.preCalc = preCalc
        self.inverseType = inverseType
        self.toggleCV = toggleCV
        self.log = log
        self.stride = 3

    # ------------------------------------------------------------------ #
    # Internal: pre-compute Xdagger on GPU                                #
    # ------------------------------------------------------------------ #

    def __preCalc_gpu(self):
        inverseType_gpu = ["CA", "GJ"]
        if self.inverseType not in inverseType_gpu:
            raise ValueError(f'inverseType must be one of {inverseType_gpu}')

        St_host = np.ascontiguousarray(self.mc.St.flatten(), dtype=np.float32)
        X_big_T    = np.zeros((self.stride * self.mc.nPeriod, self.mc.nPath), dtype=np.float32)
        Xdagger_big = np.zeros_like(X_big_T)

        St_dev         = alloc_and_copy(St_host)
        X_big_T_dev    = alloc_empty(X_big_T.nbytes)
        Xdagger_big_dev = alloc_empty(Xdagger_big.nbytes)
        # zero-init write buffers
        cuda.memset_d8(X_big_T_dev,    0, X_big_T.nbytes)
        cuda.memset_d8(Xdagger_big_dev, 0, Xdagger_big.nbytes)

        if self.inverseType == "GJ":
            src = (_KERNELS_DIR / "lsmc/knl_src_pre_calc_GaussJordan.cu").read_text()
            fn_name = 'preCalcAll_GaussJordan'
        else:
            src = (_KERNELS_DIR / "lsmc/knl_src_pre_calc_ClassicAdjoint.cu").read_text()
            fn_name = 'preCalcAll_ClassicAdjoint'

        mod = SourceModule(src % (self.mc.nPath, self.mc.nPeriod))
        fn = mod.get_function(fn_name)

        BLOCK = 256
        grid = ((self.mc.nPeriod + BLOCK - 1) // BLOCK, 1, 1)
        fn(St_dev, np.float32(self.mc.K), np.int32(self.mc.opt),
           X_big_T_dev, Xdagger_big_dev,
           block=(BLOCK, 1, 1), grid=grid)
        CUDAEnv.synchronize()

        copy_to_host(X_big_T,    X_big_T_dev)
        copy_to_host(Xdagger_big, Xdagger_big_dev)

        for buf in [St_dev, X_big_T_dev, Xdagger_big_dev]:
            buf.free()

        return Xdagger_big, X_big_T

    def __preCalc_gpu_optimized(self):
        St_host = np.ascontiguousarray(self.mc.St.flatten(), dtype=np.float32)
        X_big_T    = np.zeros((self.stride * self.mc.nPeriod, self.mc.nPath), dtype=np.float32)
        Xdagger_big = np.zeros_like(X_big_T)

        St_dev         = alloc_and_copy(St_host)
        X_big_T_dev    = alloc_empty(X_big_T.nbytes)
        Xdagger_big_dev = alloc_empty(Xdagger_big.nbytes)
        cuda.memset_d8(X_big_T_dev,    0, X_big_T.nbytes)
        cuda.memset_d8(Xdagger_big_dev, 0, Xdagger_big.nbytes)

        src = (_KERNELS_DIR / "lsmc/knl_src_pre_calc_optimized.cu").read_text()
        mod = SourceModule(src % (self.mc.nPath, self.mc.nPeriod),
                           options=["--use_fast_math"])
        fn = mod.get_function('preCalcAll_Optimized')

        BLOCK = 256
        grid = ((self.mc.nPeriod + BLOCK - 1) // BLOCK, 1, 1)
        fn(St_dev, np.float32(self.mc.K), np.int32(self.mc.opt),
           X_big_T_dev, Xdagger_big_dev,
           block=(BLOCK, 1, 1), grid=grid)
        CUDAEnv.synchronize()

        copy_to_host(X_big_T,    X_big_T_dev)
        copy_to_host(Xdagger_big, Xdagger_big_dev)

        for buf in [St_dev, X_big_T_dev, Xdagger_big_dev]:
            buf.free()

        return Xdagger_big, X_big_T

    # ------------------------------------------------------------------ #
    # Main solver                                                          #
    # ------------------------------------------------------------------ #

    def longstaff_schwartz_itm_path_fast_hybrid(self):
        start = time.perf_counter()
        dc_cashflow_t = []
        coef_t = []

        dt = self.mc.T / self.mc.nPeriod
        df = np.exp(-self.mc.r * dt)

        payoffs = self.mc.getPayoffs()
        itm_allPeriod = (payoffs != 0).sum(0)
        dc_cashflow = payoffs[:, -1]

        if self.preCalc is None:
            Xdagger_big, X_big_T = self.__preCalc_gpu()
        elif self.preCalc == 'optimized':
            Xdagger_big, X_big_T = self.__preCalc_gpu_optimized()

        for t in range(self.mc.nPeriod-2, -1, -1):
            if self.log == 'INFO':
                print('\ntime t:', t+1)
            dc_cashflow = dc_cashflow * df
            itm = payoffs[:, t].nonzero()
            num_itm = itm_allPeriod[t]

            X = X_big_T[t*self.stride : t*self.stride+self.stride, :num_itm].T
            Xdagger = Xdagger_big[t*self.stride : t*self.stride+self.stride, :num_itm]
            Y = dc_cashflow[itm].reshape(num_itm, 1)
            coef_ = Xdagger @ Y
            coef_t.append(coef_.flatten())
            cont_value = X @ coef_

            if self.log == 'INFO':
                print('X:\n', X)
                print('Y:', Y.flatten())
                print('Xdagger:\n', Xdagger)
                print('coef:', coef_.flatten())
                print('pre ds cf itm:', dc_cashflow[itm])
                print('cont val     :', cont_value.flatten())
                print('exer val     :', payoffs[:, t][itm])

            if self.toggleCV == 'ON':
                BS_itm = self.mc.BS[itm, t].flatten()
                max_CvBS = np.maximum(cont_value.flatten(), BS_itm)
                dc_cashflow[itm] = np.where(payoffs[:, t][itm] > max_CvBS, payoffs[:,t][itm], dc_cashflow[itm])
            elif self.toggleCV == 'OFF':
                dc_cashflow[itm] = np.where(payoffs[:, t][itm] > cont_value.flatten(), payoffs[:,t][itm], dc_cashflow[itm])

            dc_cashflow_t.append(dc_cashflow)
            if self.log == 'INFO':
                print('post ds cf itm:', dc_cashflow)

        C_hat = np.sum(dc_cashflow) * df / self.mc.nPath
        elapse = (time.perf_counter() - start) * 1e3
        print(f'Longstaff {CUDAEnv.deviceName} price: {C_hat} - {elapse:.3f} ms')
        return C_hat, elapse


# backward-compatible alias
LSMC_OpenCL = LSMC_CUDA


def main():
    print("longstaff.py")


if __name__ == "__main__":
    main()
