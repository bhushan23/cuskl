#pragma once
#include "linalg/custom_accum.h"
#include "linalg/gemm.h"
#include "linalg/row_gemm.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void l1Impl(int m, int n, int k,
            IType const* pA,
            IType const* pB,
            OType const* pC,
            OType* pD,
            OType alpha,
            OType beta,
            cudaStream_t stream=0)
{
  typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_t;
  typedef cutlass::Shape<1, 4, 8> ThreadsPerWarp_t;
  typedef cutlass::gemm::LinearScaling<OType> EpilogueFunctor_t;
  typedef LinAlg::ThreadL1NormAdd<AccumulatorsPerThread_t,
          ThreadsPerWarp_t, IType, IType, AccType> MainLoopFunctor_t;

  LinAlg::row_gemm<IType, AccType, OType,
    OutputTile_, EpilogueFunctor_t, AccumulatorsPerThread_t, MainLoopFunctor_t>
    (CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, pA, pB, beta, pC, pD);
}

}
}