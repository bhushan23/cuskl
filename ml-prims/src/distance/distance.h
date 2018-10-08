#pragma once

#include "cuda_utils.h"
#include "distance/cosine.h"
#include "distance/euclidean.h"
#include "distance/l1.h"
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

/** enum to tell how to compute euclidean distance */
enum DistanceType {
    /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
    EucExpandedL2 = 0,
    /** same as above, but inside the epilogue, perform square root operation */
    EucExpandedL2Sqrt,
    /** cosine distance */
    EucExpandedCosine,
    /** L1 distance */
    EucUnexpandedL1,
    /** evaluate as dist_ij += (x_ik - y-jk)^2 */
    EucUnexpandedL2,
    /** same as above, but inside the epilogue, perform square root operation */
    EucUnexpandedL2Sqrt,
};

/**
 * @brief Evaluate pairwise distances
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutParams output parameter type. It could represent simple C-like struct
 * to pass extra outputs after the computation.
 * @tparam InParams input parameter type. It could represent simple C-like struct
 * to hold extra input that might be needed during the computation.
 * @param dist output parameters
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param inParams extra input parameters
 * @param type which distance to evaluate
 * @param workspace temporary workspace needed for computations
 * @param workspaceSize number of bytes of the workspace
 * @param stream cuda stream
 *
 * @note if workspace is passed as nullptr, this will return in
 *  workspaceSize, the number of bytes of workspace required
 */
template <typename InType, typename AccType, typename OutParams,
          typename InParams, typename OutputTile_>
void distance(OutParams& dist, InType* x, InType* y, int m, int n, int k,
              InParams& inParams, DistanceType type,
              void* workspace, size_t& workspaceSize, cudaStream_t stream=0) {

    if(workspace == nullptr && type <= EucExpandedCosine) {
        workspaceSize = m * sizeof(AccType);
        if(x != y)
            workspaceSize += n * sizeof(AccType);
        return;
    }

    ///@todo: implement the distance matrix computation here
    switch(type) {
    case EucExpandedL2:
        euclideanAlgo1<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0,
          (AccType*)workspace, workspaceSize, false);
        break;
    case EucExpandedL2Sqrt:
        euclideanAlgo1<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0,
          (AccType*)workspace, workspaceSize, true);
        break;
    case EucUnexpandedL2:
        euclideanAlgo2<InType, AccType, AccType, OutputTile_>
        (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0);
        break;
    case EucUnexpandedL2Sqrt:
        euclideanAlgo2<InType, AccType, AccType, OutputTile_>
        (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0, true);
        break;
    case EucUnexpandedL1:
        l1Impl<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0, stream);
        break;
    case EucExpandedCosine:
        cosineAlgo1<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, dist.dist, dist.dist, (AccType)1, (AccType)0,
          (AccType*)workspace, workspaceSize, stream);
        break;
    default:
        ASSERT(false, "Invalid DistanceType '%d'!", type);
    };
}

}; // end namespace Distance
}; // end namespace MLCommon
