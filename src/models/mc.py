import numpy as np
from pathlib import Path
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from .utils import CUDAEnv, alloc_and_copy, alloc_empty, copy_to_host, grid1d
import time
import warnings

from scipy.stats import norm

_MODELS_DIR = Path(__file__).parent
_KERNELS_DIR = _MODELS_DIR / "kernels"


def BlackScholes(S0, K, r, sigma, T, opttype='P'):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    put_price = call_price - S0 + np.exp(-r*T)*K
    if opttype == 'C':
        price = call_price
    elif opttype == 'P':
        price = put_price
    return price


def BlackScholes_matrix(St, K, r, sigma, T, nPeriod, opttype='P'):
    BS = np.zeros_like(St, dtype=np.float32)
    dt = T / nPeriod
    for t in range(nPeriod):
        new_T = dt * (nPeriod - t)
        BS[:,t] = BlackScholes(St[:,t], K, r, sigma, new_T, 'P')
    return BS


class MonteCarloBase:
    __seed = 1001

    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.nPath = nPath
        self.nPeriod = nPeriod
        self.K = K
        self.opttype = opttype
        self.opt = None
        match self.opttype:
            case 'C':
                self.opt = -1
            case 'P':
                self.opt = 1
        self.dt = self.T / self.nPeriod
        self.Z = self.__getZ()
        self.St = self.__getSt()
        self.BS = self.__getBlackScholes_matrix()

    @classmethod
    def getSeed(cls):
        return cls.__seed

    @classmethod
    def setSeed(cls, seed):
        cls.__seed = seed

    def __getZ(self):
        if self.__seed is np.nan:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=self.__seed)
        return rng.normal(size=(self.nPath, self.nPeriod)).astype(np.float32)

    def __getSt(self):
        nudt   = (self.r - 0.5 * self.sigma**2) * self.dt
        volsdt = self.sigma * np.sqrt(self.dt)
        lnS0   = np.log(self.S0)
        lnSt   = lnS0 + np.cumsum(nudt + volsdt * self.Z, axis=1)
        return np.exp(lnSt).astype(np.float32)

    def __getBlackScholes_matrix(self):
        BS = np.zeros_like(self.St, dtype=np.float32)
        dt = self.T / self.nPeriod
        for t in range(self.nPeriod):
            new_T = dt * (self.nPeriod - t)
            BS[:,t] = BlackScholes(self.St[:,t], self.K, self.r, self.sigma, new_T, self.opttype)
        return BS

    def getPayoffs(self):
        return np.maximum(0, (self.K - self.St) * self.opt)


# --------------------------------------------------------------------------- #
# GPU-accelerated Monte Carlo                                                  #
# --------------------------------------------------------------------------- #

class hybridMonteCarlo(MonteCarloBase):
    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish):
        super().__init__(S0, r, sigma, T, nPath, nPeriod, K, opttype)
        self.nFish = nFish

        # upload Z to device (shared with PSO classes)
        self.Z_d = alloc_and_copy(self.Z)

        # PSO initialisation arrays (shared with PSO child classes)
        nDim = self.nPeriod
        rng = np.random.default_rng(seed=52)
        self.pos_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * 100.0
        self.vel_init = rng.uniform(size=(nDim, nFish)).astype(np.float32) * 5.0
        self.r1 = rng.uniform(size=(nDim, nFish)).astype(np.float32)
        self.r2 = rng.uniform(size=(nDim, nFish)).astype(np.float32)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    # ------------------------------------------------------------------ #
    # CPU baseline                                                         #
    # ------------------------------------------------------------------ #

    def getEuroOption_np(self):
        start = time.perf_counter()
        C_hat_Euro = (np.exp(-self.r*self.T)
                      * np.maximum(0, (self.K - self.St[:, -1]) * self.opt)
                      ).sum() / self.nPath
        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo Numpy European price: {C_hat_Euro} - {elapse:.3f} ms")
        return C_hat_Euro, elapse

    # ------------------------------------------------------------------ #
    # GPU: optimised one-shot  <-- WINNER                                  #
    # ------------------------------------------------------------------ #

    def getEuroOption_cl_optimized(self):
        start = time.perf_counter()
        kernel_src = (_KERNELS_DIR / "mc/knl_source_mc_getEuroOption.cu").read_text()
        mod = SourceModule(kernel_src % (self.nPath, self.nPeriod),
                           options=["--use_fast_math"])
        fn = mod.get_function("getEuroOption_optimized")

        payoffs   = np.empty(self.nPath, dtype=np.float32)
        payoffs_d = alloc_empty(payoffs.nbytes)

        block, grid = grid1d(self.nPath)
        fn(self.Z_d,
           np.float32(np.log(self.S0)),
           np.float32(self.K),
           np.float32(self.r),
           np.float32(self.sigma),
           np.float32(self.T),
           np.int32(self.opt),
           np.float32(np.exp(-self.r * self.T)),
           payoffs_d,
           block=block, grid=grid)

        CUDAEnv.synchronize()
        copy_to_host(payoffs, payoffs_d)
        payoffs_d.free()

        C_hat_Euro = payoffs.sum() / self.nPath
        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo {CUDAEnv.deviceName} European price: {C_hat_Euro} - {elapse:.3f} ms")
        return C_hat_Euro, elapse

    # ------------------------------------------------------------------ #
    # GPU: two-pass reduction sum                                          #
    # ------------------------------------------------------------------ #

    def getEuroOption_cl_optimize_reductionSum(self):
        CEILING = 65536
        if self.nPath > CEILING:
            warnings.warn(
                f"getEuroOption_cl_optimize_reductionSum only works up to {CEILING} paths.",
                UserWarning, stacklevel=2)

        start = time.perf_counter()
        kernel_src = (_KERNELS_DIR / "mc/knl_source_mc_getEuroOption.cu").read_text()
        mod = SourceModule(kernel_src % (self.nPath, self.nPeriod),
                           options=["--use_fast_math"])
        fn_sum1 = mod.get_function("getEuroOption_optimized_sum1")
        fn_sum2 = mod.get_function("getEuroOption_optimized_sum2")

        BLOCK_SIZE = 256
        n_groups = (self.nPath + BLOCK_SIZE - 1) // BLOCK_SIZE

        C_hats   = np.empty(n_groups, dtype=np.float32)
        C_hats_d = alloc_empty(C_hats.nbytes)

        fn_sum1(self.Z_d,
                np.float32(np.log(self.S0)),
                np.float32(self.K),
                np.float32(self.r),
                np.float32(self.sigma),
                np.float32(self.T),
                np.int32(self.opt),
                np.float32(np.exp(-self.r * self.T)),
                C_hats_d,
                block=(BLOCK_SIZE, 1, 1),
                grid=(n_groups, 1, 1),
                shared=BLOCK_SIZE * 4)

        final_result   = np.empty(1, dtype=np.float32)
        final_result_d = alloc_empty(final_result.nbytes)
        block2 = min(n_groups, BLOCK_SIZE)
        fn_sum2(np.int32(n_groups),
                C_hats_d,
                final_result_d,
                block=(block2, 1, 1),
                grid=(1, 1, 1),
                shared=block2 * 4)

        CUDAEnv.synchronize()
        copy_to_host(final_result, final_result_d)
        C_hat_Euro = final_result.sum() / self.nPath

        C_hats_d.free()
        final_result_d.free()

        elapse = (time.perf_counter() - start) * 1e3
        print(f"MonteCarlo {CUDAEnv.deviceName}-reductionSum European price: {C_hat_Euro} - {elapse:.3f} ms")
        return C_hat_Euro

    def cleanUp(self):
        self.Z_d.free()


def main():
    S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 10, 3, 100.0, 'P', 500
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
    print(mc.St)


if __name__ == "__main__":
    main()
