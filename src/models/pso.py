import numpy as np
from pathlib import Path
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from .mc import MonteCarloBase
import matplotlib.pyplot as plt
from .utils import CUDAEnv, alloc_and_copy, alloc_empty, copy_to_host, copy_to_device, grid1d
import time

_MODELS_DIR = Path(__file__).parent
_KERNELS_DIR = _MODELS_DIR / "kernels"

# default block size for PSO launches (each thread = one fish)
_PSO_BLOCK = 128


def _pso_launch(nFish):
    """Return (block, grid) for a 1-D PSO kernel over nFish threads."""
    blk = min(nFish, _PSO_BLOCK)
    grd = (nFish + blk - 1) // blk
    return (blk, 1, 1), (grd, 1, 1)


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class PSOBase:
    _w  = 0.5
    _c1 = 0.5
    _c2 = 0.5
    _criteria = 1e-6

    def __init__(self, mc: MonteCarloBase, nFish):
        self.mc    = mc
        self.nDim  = self.mc.nPeriod
        self.nFish = nFish
        self.dt    = self.mc.T / self.mc.nPeriod


# --------------------------------------------------------------------------- #
# CPU-only PSO (unchanged)                                                    #
# --------------------------------------------------------------------------- #

class PSO_Numpy(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        self.fitFunc_vectorized = np.vectorize(self._costPsoAmerOption_np, signature='(n)->()')
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.costs      = np.zeros((nFish,), dtype=np.float32)
        self.pbest_costs = self.costs.copy()
        self.pbest_pos   = self.position.copy()
        gid = np.argmax(self.pbest_costs)
        self.gbest_cost = self.pbest_costs[gid]
        self.gbest_pos  = self.pbest_pos[:, gid]
        self.BestCosts  = np.array([])

    def _searchGrid(self):
        self.velocity = (self._w * self.velocity
                         + self._c1 * self.r1 * (self.pbest_pos - self.position)
                         + self._c2 * self.r2 * (self.gbest_pos.reshape(self.nDim, 1) - self.position))
        self.position += self.velocity

    def _costPsoAmerOption_np(self, in_particle):
        crossings   = self.mc.St < in_particle[None, :]
        has_crossing = np.any(crossings, axis=1)
        boundaryIdx = np.argmax(crossings, axis=1)
        boundaryIdx[~has_crossing] = self.mc.nPeriod - 1
        exerciseSt = self.mc.St[np.arange(len(boundaryIdx)), boundaryIdx]
        searchCost = (np.exp(-self.mc.r * (boundaryIdx+1) * self.dt)
                      * np.maximum(0, (self.mc.K - exerciseSt)*self.mc.opt)).sum() / self.mc.nPath
        return searchCost

    def solvePsoAmerOption_np(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()
        for i in range(self.iterMax):
            t = time.perf_counter()
            self._searchGrid()
            search.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            self.costs = self.fitFunc_vectorized(np.transpose(self.position)).astype(np.float32)
            fit.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            mask = np.greater(self.costs, self.pbest_costs)
            self.pbest_costs[mask] = self.costs[mask]
            self.pbest_pos[:, mask] = self.position[:, mask]
            gid = np.argmax(self.pbest_costs)
            if self.pbest_costs[gid] > self.gbest_cost:
                self.gbest_cost = self.pbest_costs[gid]
                self.gbest_pos  = self.pbest_pos[:, gid]
            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            rest.append((time.perf_counter()-t)*1e3)

            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1]-self.BestCosts[-2]) < self._criteria:
                break

        C_hat  = self.gbest_cost
        elapse = (time.perf_counter()-start)*1e3
        print(f'Pso numpy price    : {C_hat} - {elapse:.3f} ms')
        return C_hat, elapse, search, fit, rest


# --------------------------------------------------------------------------- #
# GPU PSO – hybrid (search on GPU, pbest/gbest on CPU)                        #
# --------------------------------------------------------------------------- #

class PSO_CUDA_hybrid(PSOBase):
    """
    CUDA port of PSO_OpenCL_hybrid.
    searchGrid + American-option fitness run on GPU;
    pbest / gbest updates run on CPU.
    """

    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax

        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2

        self.pos_d  = alloc_and_copy(self.position)
        self.vel_d  = alloc_and_copy(self.velocity)
        self.r1_d   = alloc_and_copy(self.r1)
        self.r2_d   = alloc_and_copy(self.r2)
        self.St_d   = alloc_and_copy(self.mc.St)

        self.boundary_idx = np.empty((self.mc.nPath, nFish), dtype=np.int32)
        self.exercise     = np.empty((self.mc.nPath, nFish), dtype=np.float32)
        self.boundary_idx_d = alloc_and_copy(self.boundary_idx)
        self.exercise_d     = alloc_and_copy(self.exercise)

        self.costs      = np.zeros((nFish,), dtype=np.float32)
        self.costs_d    = alloc_empty(self.costs.nbytes)

        self.pbest_costs = self.costs.copy()
        self.pbest_pos   = self.position.copy()
        self.pbest_pos_d = alloc_and_copy(self.pbest_pos)

        gid = np.argmax(self.pbest_costs)
        self.gbest_cost = self.pbest_costs[gid]
        self.gbest_pos  = self.pbest_pos[:, gid].copy()
        self.gbest_pos_d = alloc_and_copy(self.gbest_pos)

        self.BestCosts = np.array([])

        # compile kernels
        src_sg = (_KERNELS_DIR / "pso/scalar/knl_source_pso_searchGrid.cu").read_text()
        mod_sg = SourceModule(src_sg % self.nDim)
        self.knl_searchGrid = mod_sg.get_function("searchGrid")

        src_ao = (_KERNELS_DIR / "pso/scalar/knl_source_pso_getAmerOption.cu").read_text()
        mod_ao = SourceModule(src_ao % (self.mc.nPath, self.mc.nPeriod),
                              options=["--use_fast_math"])
        self.knl_psoAmerOption_gb = mod_ao.get_function("psoAmerOption_gb")

    def _searchGrid(self):
        block, grid = _pso_launch(self.nFish)
        self.knl_searchGrid(
            self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d,
            self.r1_d, self.r2_d,
            np.float32(self._w), np.float32(self._c1), np.float32(self._c2),
            np.int32(self.nFish),
            block=block, grid=grid)
        CUDAEnv.synchronize()

    def _costPsoAmerOption_cl(self):
        block, grid = _pso_launch(self.nFish)
        self.knl_psoAmerOption_gb(
            self.St_d, self.pos_d, self.costs_d,
            self.boundary_idx_d, self.exercise_d,
            np.float32(self.mc.r), np.float32(self.mc.T),
            np.float32(self.mc.K), np.int32(self.mc.opt),
            np.int32(self.nFish),
            block=block, grid=grid)
        CUDAEnv.synchronize()

    def solvePsoAmerOption_cl(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()

        for i in range(self.iterMax):
            t = time.perf_counter()
            self._searchGrid()
            copy_to_host(self.position, self.pos_d)
            search.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            self._costPsoAmerOption_cl()
            copy_to_host(self.costs, self.costs_d)
            fit.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            mask = np.greater(self.costs, self.pbest_costs)
            self.pbest_costs[mask] = self.costs[mask]
            self.pbest_pos[:, mask] = self.position[:, mask]
            copy_to_device(self.pbest_pos_d, self.pbest_pos)

            gid = np.argmax(self.pbest_costs)
            if self.pbest_costs[gid] > self.gbest_cost:
                self.gbest_cost = self.pbest_costs[gid]
                self.gbest_pos  = self.pbest_pos[:, gid].copy()
                copy_to_device(self.gbest_pos_d, self.gbest_pos)

            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            rest.append((time.perf_counter()-t)*1e3)

            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1]-self.BestCosts[-2]) < self._criteria:
                break

        C_hat  = self.gbest_cost
        elapse = (time.perf_counter()-start)*1e3
        print(f'Pso cuda_hybrid price: {C_hat} - {elapse:.3f} ms')
        self.cleanUp()
        return C_hat, elapse, search, fit, rest

    def cleanUp(self):
        for d in [self.St_d, self.boundary_idx_d, self.exercise_d,
                  self.pos_d, self.vel_d, self.r1_d, self.r2_d,
                  self.costs_d, self.pbest_pos_d, self.gbest_pos_d]:
            d.free()


# --------------------------------------------------------------------------- #
# GPU PSO – scalar (separate search / fitness / pbest kernels)                #
# --------------------------------------------------------------------------- #

class PSO_CUDA_scalar(PSOBase):
    """
    CUDA port of PSO_OpenCL_scalar.
    All hot steps (searchGrid, fitness, pbest update, gbest update) run on GPU.
    """

    def __init__(self, mc: MonteCarloBase, nFish, direction='backward', iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax

        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2

        self.pos_d  = alloc_and_copy(self.position)
        self.vel_d  = alloc_and_copy(self.velocity)
        self.r1_d   = alloc_and_copy(self.r1)
        self.r2_d   = alloc_and_copy(self.r2)
        self.St_d   = alloc_and_copy(self.mc.St)

        self.costs       = np.zeros((nFish,), dtype=np.float32)
        self.costs_d     = alloc_empty(self.costs.nbytes)
        self.pbest_costs = self.costs.copy()
        self.pbest_costs_d = alloc_empty(self.pbest_costs.nbytes)
        cuda.memset_d8(self.pbest_costs_d, 0, self.pbest_costs.nbytes)

        self.pbest_pos   = self.position.copy()
        self.pbest_pos_d = alloc_and_copy(self.pbest_pos)

        gid = np.argmax(self.pbest_costs)
        self.gbest_cost = self.pbest_costs[gid]
        self.gbest_pos  = self.pbest_pos[:, gid].copy()
        self.gbest_pos_d = alloc_and_copy(self.gbest_pos)

        self.BestCosts = np.array([])

        # compile kernels
        src_sg = (_KERNELS_DIR / "pso/scalar/knl_source_pso_searchGrid.cu").read_text()
        mod_sg = SourceModule(src_sg % self.nDim)
        self.knl_searchGrid = mod_sg.get_function("searchGrid")

        src_ao = (_KERNELS_DIR / "pso/scalar/knl_source_pso_getAmerOption.cu").read_text()
        mod_ao = SourceModule(src_ao % (self.mc.nPath, self.mc.nPeriod),
                              options=["--use_fast_math"])
        if direction == 'forward':
            self.knl_psoAmerOption_gb = mod_ao.get_function("psoAmerOption_gb2")
        else:
            self.knl_psoAmerOption_gb = mod_ao.get_function("psoAmerOption_gb3")

        src_ub = (_KERNELS_DIR / "pso/scalar/knl_source_pso_updateBests.cu").read_text()
        mod_ub = SourceModule(src_ub % (self.nDim, self.nFish))
        self.knl_update_pbest    = mod_ub.get_function("update_pbest")
        self.knl_update_gbest_pos = mod_ub.get_function("update_gbest_pos")

    def _searchGrid(self):
        block, grid = _pso_launch(self.nFish)
        self.knl_searchGrid(
            self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d,
            self.r1_d, self.r2_d,
            np.float32(self._w), np.float32(self._c1), np.float32(self._c2),
            np.int32(self.nFish),
            block=block, grid=grid)
        CUDAEnv.synchronize()

    def _costPsoAmerOption_cl(self):
        block, grid = _pso_launch(self.nFish)
        self.knl_psoAmerOption_gb(
            self.St_d, self.pos_d, self.costs_d,
            np.float32(self.mc.r), np.float32(self.mc.T),
            np.float32(self.mc.K), np.int32(self.mc.opt),
            np.int32(self.nFish),
            block=block, grid=grid)
        CUDAEnv.synchronize()

    def solvePsoAmerOption_cl(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()

        for i in range(self.iterMax):
            t = time.perf_counter()
            self._searchGrid()
            search.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            self._costPsoAmerOption_cl()
            fit.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            block, grid = _pso_launch(self.nFish)
            self.knl_update_pbest(
                self.costs_d, self.pbest_costs_d,
                self.pos_d, self.pbest_pos_d,
                np.int32(self.nFish),
                block=block, grid=grid)
            copy_to_host(self.pbest_costs, self.pbest_costs_d)
            CUDAEnv.synchronize()

            gid = np.argmax(self.pbest_costs)
            if self.pbest_costs[gid] > self.gbest_cost:
                self.gbest_cost = self.pbest_costs[gid]
                dim_block = (min(self.nDim, 256), 1, 1)
                dim_grid  = ((self.nDim + 255) // 256, 1, 1)
                self.knl_update_gbest_pos(
                    self.gbest_pos_d, self.pbest_pos_d, np.int32(gid),
                    block=dim_block, grid=dim_grid)
                CUDAEnv.synchronize()

            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            rest.append((time.perf_counter()-t)*1e3)

            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1]-self.BestCosts[-2]) < self._criteria:
                break

        C_hat  = self.gbest_cost
        elapse = (time.perf_counter()-start)*1e3
        print(f'Pso cuda_scalar price: {C_hat} - {elapse:.3f} ms')
        self.cleanUp()
        return C_hat, elapse, search, fit, rest

    def cleanUp(self):
        for d in [self.St_d, self.pos_d, self.vel_d, self.r1_d, self.r2_d,
                  self.costs_d, self.pbest_costs_d, self.pbest_pos_d, self.gbest_pos_d]:
            d.free()


# --------------------------------------------------------------------------- #
# GPU PSO – fused scalar (single fused kernel per iteration)                  #
# --------------------------------------------------------------------------- #

class PSO_CUDA_scalar_fusion(PSOBase):
    """
    CUDA port of PSO_OpenCL_scalar_fusion.
    Fused kernel does searchGrid + fitness + pbest update in one GPU call.
    """

    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax

        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2

        self.pos_d  = alloc_and_copy(self.position)
        self.vel_d  = alloc_and_copy(self.velocity)
        self.r1_d   = alloc_and_copy(self.r1)
        self.r2_d   = alloc_and_copy(self.r2)
        self.St_d   = alloc_and_copy(self.mc.St)

        self.pbest_costs = np.zeros((nFish,), dtype=np.float32)
        self.pbest_costs_d = alloc_empty(self.pbest_costs.nbytes)
        cuda.memset_d8(self.pbest_costs_d, 0, self.pbest_costs.nbytes)

        self.pbest_pos   = self.position.copy()
        self.pbest_pos_d = alloc_and_copy(self.pbest_pos)

        gid = np.argmax(self.pbest_costs)
        self.gbest_cost  = self.pbest_costs[gid]
        self.gbest_pos   = self.pbest_pos[:, gid].copy()
        self.gbest_pos_d = alloc_and_copy(self.gbest_pos)

        self.BestCosts = np.array([])

        # compile fused kernel
        src = (_KERNELS_DIR / "pso/scalar/knl_source_pso_fusion.cu").read_text()
        mod = SourceModule(src % (self.nDim, self.mc.nPath, self.mc.nPeriod, self.nFish),
                           options=["--use_fast_math"])
        self.knl_pso             = mod.get_function("pso")
        self.knl_update_gbest_pos = mod.get_function("update_gbest_pos")

    def _runPso(self):
        block, grid = _pso_launch(self.nFish)
        self.knl_pso(
            self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d,
            self.r1_d, self.r2_d,
            np.float32(self._w), np.float32(self._c1), np.float32(self._c2),
            self.St_d,
            np.float32(self.mc.r), np.float32(self.mc.T),
            np.float32(self.mc.K), np.int32(self.mc.opt),
            self.pbest_costs_d,
            np.int32(self.nFish),
            block=block, grid=grid)
        CUDAEnv.synchronize()

    def solvePsoAmerOption_cl(self):
        search_fit, rest = [], []
        start = time.perf_counter()

        for i in range(self.iterMax):
            t = time.perf_counter()
            self._runPso()
            copy_to_host(self.pbest_costs, self.pbest_costs_d)
            search_fit.append((time.perf_counter()-t)*1e3)

            t = time.perf_counter()
            gid = np.argmax(self.pbest_costs)
            if self.pbest_costs[gid] > self.gbest_cost:
                self.gbest_cost = self.pbest_costs[gid]
                dim_block = (min(self.nDim, 256), 1, 1)
                dim_grid  = ((self.nDim + 255) // 256, 1, 1)
                self.knl_update_gbest_pos(
                    self.gbest_pos_d, self.pbest_pos_d, np.int32(gid),
                    block=dim_block, grid=dim_grid)
                CUDAEnv.synchronize()

            self.BestCosts = np.concatenate((self.BestCosts, [self.gbest_cost]))
            rest.append((time.perf_counter()-t)*1e3)

            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1]-self.BestCosts[-2]) < self._criteria:
                break

        C_hat  = self.gbest_cost
        elapse = (time.perf_counter()-start)*1e3
        print(f'Pso cuda_scalar_fusion price: {C_hat} - {elapse:.3f} ms')
        self.cleanUp()
        return C_hat, elapse, search_fit, rest

    def cleanUp(self):
        for d in [self.St_d, self.pos_d, self.vel_d, self.r1_d, self.r2_d,
                  self.pbest_costs_d, self.pbest_pos_d, self.gbest_pos_d]:
            d.free()


# backward-compatible aliases
PSO_OpenCL_hybrid        = PSO_CUDA_hybrid
PSO_OpenCL_scalar        = PSO_CUDA_scalar
PSO_OpenCL_scalar_fusion = PSO_CUDA_scalar_fusion
