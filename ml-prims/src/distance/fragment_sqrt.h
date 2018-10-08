#pragma once

#include "cuda_utils.h"
#include <cutlass/fragment_multiply_add.h>
#include <math_functions.h>

namespace cutlass {
namespace gemm {

template <typename Scalar_>
struct FragmentSqrt : public FragmentMultiplyAdd<Scalar_> {
  /// Ctor.
  CUTLASS_DEVICE FragmentSqrt() : FragmentMultiplyAdd<Scalar_>() {}

  /// d = sqrt(b).
  template <typename FragmentB_, typename FragmentCd_>
  CUTLASS_DEVICE void sqrt(FragmentB_ const& b, FragmentCd_& d) {
    int const kReduction = FragmentB_::kElements / FragmentCd_::kElements;
    for (int j = 0; j < FragmentCd_::kElements; ++j) {
      d[j] = MLCommon::mySqrt( b[j * kReduction + 0] );
      for (int k = 1; k < kReduction; ++k) {
        d[j] += MLCommon::mySqrt( b[j * kReduction + k] );
      }
    }
  }
};

}
}