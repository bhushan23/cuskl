#pragma once

#include "cuda_utils.h"
#include "linalg/gemm.h"
#include "linalg/norm.h"
#include "linalg/row_gemm.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_,
          typename Lambda>
void distanceAlgo1(int m, int n, int k,
                   IType const* pA,
                   IType const* pB,
                   OType const* pC,
                   OType* pD,
                   OType alpha,
                   OType beta,
                   AccType* pWorkspace,
                   size_t workspaceSize,
                   Lambda op,
                   cudaStream_t stream=0)
{

  if (((pA != pB) && (workspaceSize < (m + n) * sizeof(AccType))) ||
      (workspaceSize < m * sizeof(AccType))) {
    THROW("workspace size error");
  }
  if (pWorkspace == nullptr) {
    THROW("workspace is null");
  }

  LinAlg::row_gemm<IType, AccType, OType, OutputTile_>(CUBLAS_OP_N, CUBLAS_OP_T,
    m, n, k, alpha, pA, pB, beta, pC, pD);

  AccType* dotA = pWorkspace;
  AccType* dotB = pWorkspace;
  if (pA != pB) {
    dotB += m;
    LinAlg::norm(dotA, pA, k, m, LinAlg::L2Norm);
    LinAlg::norm(dotB, pB, k, n, LinAlg::L2Norm);
  }
  else {
    LinAlg::norm(dotA, pA, k, m, LinAlg::L2Norm);
  }

  op(m, n, dotA, dotB, pC, pD, alpha, beta, stream);
}

}; // end namespace Distance
}; // end namespace MLCommon
