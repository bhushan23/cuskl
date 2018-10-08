#pragma once

#include <cutlass/gemm/linear_scaling.h>
#include <distance/fragment_sqrt.h>

namespace cutlass {
namespace gemm {

template <typename Scalar_, typename FragmentMultiplyAdd_ = FragmentSqrt<Scalar_> >
struct LinearScalingSqrt : public LinearScaling<Scalar_, FragmentMultiplyAdd_> {
  // The scalar.
  typedef Scalar_ Scalar;
  // The adapater.
  typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;
  /// Ctor.
  CUTLASS_DEVICE LinearScalingSqrt(typename LinearScaling<Scalar, FragmentMultiplyAdd>::Params const& params) :
    LinearScaling<Scalar, FragmentMultiplyAdd_>(params) {}

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp;
    mad.sqrt(accum, tmp);
    mad.multiply(LinearScaling<Scalar, FragmentMultiplyAdd>::alpha, tmp, output);
  }

  /// Evaluate the functor.
  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const& old, FragmentB_& output) {
    FragmentMultiplyAdd mad;
    FragmentB_ tmp0, tmp1;
    mad.multiply(LinearScaling<Scalar, FragmentMultiplyAdd>::beta, old, tmp0);
    mad.sqrt(accum, tmp1);
    mad.multiply_add(LinearScaling<Scalar, FragmentMultiplyAdd>::alpha, tmp1, tmp0, output);
  }
};

}
}
