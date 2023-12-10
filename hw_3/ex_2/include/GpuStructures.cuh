#ifndef GPU_STRUCTS_CUH
#define GPU_STRUCTS_CUH

#include "Parameters.h"
#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(_cudaCmd)                                            \
{                                                                             \
    cudaError_t cudaStatus;                                                   \
    cudaStatus = _cudaCmd;                                                    \
    if (cudaStatus != cudaSuccess)                                            \
    {                                                                         \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));  \
    }                                                                         \
}           


//______________________________________________________________________________________
// GPU PARAMETERS

struct parametersGPU
{
    double c;
    double dt;

    bool PERIODICX;
    bool PERIODICY;
    bool PERIODICZ;
};

void initParameterGPU(struct parameters* paramCPU, struct parametersGPU* paramGPU)
{
    paramGPU->c = paramCPU->c;
    paramGPU->dt = paramCPU->dt;
    paramGPU->PERIODICX = paramCPU->PERIODICX;
    paramGPU->PERIODICY = paramCPU->PERIODICY;
    paramGPU->PERIODICZ = paramCPU->PERIODICZ;
}


//______________________________________________________________________________________
// GPU GRID

struct gridGPU
{
    double xStart, yStart, zStart;
    FPfield invdx, invdy, invdz, invVOL;
    int nxn, nyn, nzn;
    double Lx, Ly, Lz;

    FPfield* XN_flat;
    FPfield* YN_flat;
    FPfield* ZN_flat;
};

void allocateGridGPU(struct grid* grdCPU, struct gridGPU* grdGPU)
{
    // Copy grid parameters
    grdGPU->xStart = grdCPU->xStart;
    grdGPU->yStart = grdCPU->yStart;
    grdGPU->zStart = grdCPU->zStart;
    grdGPU->invdx = grdCPU->invdx;
    grdGPU->invdy = grdCPU->invdy;
    grdGPU->invdz = grdCPU->invdz;
    grdGPU->invVOL = grdCPU->invVOL;
    grdGPU->nxn = grdCPU->nxn;
    grdGPU->nyn = grdCPU->nyn;
    grdGPU->nzn = grdCPU->nzn;
    grdGPU->Lx = grdCPU->Lx;
    grdGPU->Ly = grdCPU->Ly;
    grdGPU->Lz = grdCPU->Lz;

    // Allocate flat array grid points
    CUDA_CHECK_ERROR(cudaMalloc((void**)&grdGPU->XN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&grdGPU->YN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&grdGPU->ZN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield)));

    // Copy flat array data from CPU to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(grdGPU->XN_flat, grdCPU->XN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(grdGPU->YN_flat, grdCPU->YN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(grdGPU->ZN_flat, grdCPU->ZN_flat, grdCPU->nxn * grdCPU->nyn * grdCPU->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
}

void freeGridGPU(struct gridGPU* grdGPU)
{
    CUDA_CHECK_ERROR(cudaFree(grdGPU->XN_flat));
    CUDA_CHECK_ERROR(cudaFree(grdGPU->YN_flat));
    CUDA_CHECK_ERROR(cudaFree(grdGPU->ZN_flat));
}


//______________________________________________________________________________________
// GPU FIELD

struct fieldGPU
{
    FPfield* Ex_flat;
    FPfield* Ey_flat;
    FPfield* Ez_flat;

    FPfield* Bxn_flat;
    FPfield* Byn_flat;
    FPfield* Bzn_flat;
};

void allocateFieldGPU(struct grid* grd, struct EMfield* fieldCPU, struct fieldGPU* fieldGPU)
{
    // Allocate flat array electro-magnetic field
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&fieldGPU->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)));

    // Copy flat array data from CPU to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Ex_flat, fieldCPU->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Ey_flat, fieldCPU->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Ez_flat, fieldCPU->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Bxn_flat, fieldCPU->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Byn_flat, fieldCPU->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(fieldGPU->Bzn_flat, fieldCPU->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice));
}

void freeFieldGPU(struct fieldGPU* fieldGPU)
{
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Ex_flat));
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Ey_flat));
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Ez_flat));
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Bxn_flat));
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Byn_flat));
    CUDA_CHECK_ERROR(cudaFree(fieldGPU->Bzn_flat));
}


//______________________________________________________________________________________
// GPU PARTICLES

struct particlesGPU
{
    long nop;
    int n_sub_cycles;
    int NiterMover;
    FPpart qom;

    FPpart* x; FPpart*  y; FPpart* z;
    FPpart* u; FPpart* v; FPpart* w;
};

void allocateParticleGPU(struct particles* partCPU, struct particlesGPU* partGPU)
{
    // Copy grid parameters
    partGPU->nop = partCPU->nop;
    partGPU->n_sub_cycles = partCPU->n_sub_cycles;
    partGPU->NiterMover = partCPU->NiterMover;
    partGPU->qom = partCPU->qom;

    // Allocate flat array positions and velocities
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->x, partCPU->npmax * sizeof(FPpart)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->y, partCPU->npmax * sizeof(FPpart)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->z, partCPU->npmax * sizeof(FPpart)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->u, partCPU->npmax * sizeof(FPpart)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->v, partCPU->npmax * sizeof(FPpart)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&partGPU->w, partCPU->npmax * sizeof(FPpart)));

    // Copy flat array data from CPU to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->x, partCPU->x, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->y, partCPU->y, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->z, partCPU->z, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->u, partCPU->u, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->v, partCPU->v, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(partGPU->w, partCPU->w, partCPU->npmax * sizeof(FPpart), cudaMemcpyHostToDevice));
}

void copyParticleBackToCPU(struct particlesGPU* partGPU, struct particles* partCPU)
{
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->x, partGPU->x, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->y, partGPU->y, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->z, partGPU->z, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->u, partGPU->u, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->v, partGPU->v, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(partCPU->w, partGPU->w, partCPU->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost));
}

void freeParticleGPU(struct particlesGPU* partGPU)
{
    CUDA_CHECK_ERROR(cudaFree(partGPU->x));
    CUDA_CHECK_ERROR(cudaFree(partGPU->y));
    CUDA_CHECK_ERROR(cudaFree(partGPU->z));
    CUDA_CHECK_ERROR(cudaFree(partGPU->u));
    CUDA_CHECK_ERROR(cudaFree(partGPU->v));
    CUDA_CHECK_ERROR(cudaFree(partGPU->w));
}


#endif // GPU_STRUCTS_CUH