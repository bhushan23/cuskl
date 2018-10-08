#pragma once
#include "distance/algo1.h"
#include "distance/linear_scaling_sqrt.h"
#include "linalg/custom_accum.h"
#include "linalg/eltwise2d.h"
#include "linalg/gemm.h"
#include "linalg/row_gemm.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void euclideanAlgo1(int m, int n, int k,
                   IType const* pA,
                   IType const* pB,
                   OType const* pC,
                   OType* pD,
                   OType alpha,
                   OType beta,
                   AccType* pWorkspace,
                   size_t workspaceSize,
                   bool enable_sqrt,
                   cudaStream_t stream=0)
{
  auto op = [] __device__ (OType a, OType b, OType ab) {
    return a + b - 2 * ab;
  };

  auto lambda = [=] (int rows, int cols,
      const OType* dotA, const OType* dotB, const OType* pC, OType* pD,
      OType alpha, OType beta,
      cudaStream_t stream) {
    LinAlg::eltwise2D<OType>(rows, cols, dotA, dotB, pC, pD,
      alpha, beta, op, stream);
  };

  auto sqrt_op = [] __device__ (OType a, OType b, OType ab) {
    return sqrt(a + b - 2 * ab);
  };

  auto sqrt_lambda = [=] (int rows, int cols,
      const OType* dotA, const OType* dotB, const OType* pC, OType* pD,
      OType alpha, OType beta,
      cudaStream_t stream) {
    LinAlg::eltwise2D<OType>(rows, cols, dotA, dotB, pC, pD,
        alpha, beta, sqrt_op, stream);
  };

  if (enable_sqrt) {
    distanceAlgo1<IType, AccType, OType, OutputTile_>(m, n, k,
      pA, pB, pC, pD,
      alpha, beta,
      pWorkspace, workspaceSize,
      sqrt_lambda,
      stream);
  }
  else {
    distanceAlgo1<IType, AccType, OType, OutputTile_>(m, n, k,
      pA, pB, pC, pD,
      alpha, beta,
      pWorkspace, workspaceSize,
      lambda,
      stream);
  }
}

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void euclideanAlgo2(int m, int n, int k,
                   IType const* pA,
                   IType const* pB,
                   OType const* pC,
                   OType* pD,
                   OType alpha,
                   OType beta,
                   bool enable_sqrt = false)
{
  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_t;
  typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp_t;
  typedef cutlass::gemm::LinearScaling<OType> EpilogueFunctor_t;
  typedef cutlass::gemm::LinearScalingSqrt<OType> SqrtEpilogueFunctor_t;
  typedef LinAlg::ThreadDiffSquaredAdd<AccumulatorsPerThread_t,
          ThreadsPerWarp_t, IType, IType, AccType> MainLoopFunctor_t;

  if (enable_sqrt) {
    LinAlg::row_gemm<IType, AccType, OType,
      OutputTile_, SqrtEpilogueFunctor_t, AccumulatorsPerThread_t, MainLoopFunctor_t>
      (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
  }
  else {
    LinAlg::row_gemm<IType, AccType, OType,
      OutputTile_, EpilogueFunctor_t, AccumulatorsPerThread_t, MainLoopFunctor_t>
      (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
  }
}

}
}
