#pragma once
#include "distance/algo1.h"
#include "linalg/eltwise2d.h"

namespace MLCommon {
namespace Distance {

template <typename IType,
          typename AccType,
          typename OType,
          typename OutputTile_>
void cosineAlgo1(int m, int n, int k,
                 IType const* pA,
                 IType const* pB,
                 OType const* pC,
                 OType* pD,
                 OType alpha,
                 OType beta,
                 AccType* pWorkspace,
                 size_t workspaceSize,
                 cudaStream_t stream=0)
{
  auto op = [] __device__ (OType a, OType b, OType ab) {
    return ab / (sqrt(a) * sqrt(b));
  };

  auto lambda = [=] (int rows, int cols,
      const OType* dotA, const OType* dotB, const OType* pC, OType* pD,
      OType alpha, OType beta,
      cudaStream_t stream) {
    LinAlg::eltwise2D<OType>(m, n, dotA, dotB, pC, pD, alpha, beta,
      op, stream);
  };

  distanceAlgo1<IType, AccType, OType, OutputTile_>(m, n, k,
    pA, pB, pC, pD,
    alpha, beta,
    pWorkspace, workspaceSize,
    lambda, stream);
}

}
}
