#include <gtest/gtest.h>
#include "distance/distance.h"
#include "test_utils.h"
#include "random/rng.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace Distance {

template <typename Type>
__global__ void naiveDistanceKernel(Type* out, const Type* x, const Type* y,
                                    int m, int n, int k, DistanceType type) {
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n)
        return;
    Type acc = Type(0);
    for(int i=0; i<k; ++i) {
        auto diff = x[i + midx * k] - y[i + nidx * k];
        acc += diff * diff;
    }
    if(type == EucExpandedL2Sqrt || type == EucUnexpandedL2Sqrt)
        acc = mySqrt(acc);
    out[midx * n + nidx] = acc;
}

template <typename Type>
__global__ void naiveL1DistanceKernel(
    Type* out, const Type* x, const Type* y,
    int m, int n, int k)
{
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n) {
        return;
    }

    Type acc = Type(0);
    for(int i = 0; i < k; ++i) {
        auto a = x[i + midx * k];
        auto b = y[i + nidx * k];
        auto diff = (a > b) ? (a - b) : (b - a);
        acc += diff;
    }

    out[midx * n + nidx] = acc;
}

template <typename Type>
__global__ void naiveCosineDistanceKernel(
    Type* out, const Type* x, const Type* y,
    int m, int n, int k)
{
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n) {
        return;
    }

    Type acc_a  = Type(0);
    Type acc_b  = Type(0);
    Type acc_ab = Type(0);

    for(int i = 0; i < k; ++i) {
        auto a = x[i + midx * k];
        auto b = y[i + nidx * k];

        acc_a  += a * a;
        acc_b  += b * b;
        acc_ab += a * b;
    }

    out[midx * n + nidx] = acc_ab / (sqrt(acc_a) * sqrt(acc_b));
}

template <typename Type>
void naiveDistance(Type* out, const Type* x, const Type* y, int m, int n, int k,
                   DistanceType type) {
    static const dim3 TPB(16, 32, 1);
    dim3 nblks(ceildiv(m, (int)TPB.x), ceildiv(n, (int)TPB.y), 1);

    switch (type) {
        case EucUnexpandedL1:
            naiveL1DistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k);
            break;
        case EucUnexpandedL2Sqrt:
        case EucUnexpandedL2:
        case EucExpandedL2Sqrt:
        case EucExpandedL2:
            naiveDistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k, type);
            break;
        case EucExpandedCosine:
            naiveCosineDistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k);
            break;
        default:
            FAIL() << "should be here\n";
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct DistanceInputs {
    T tolerance;
    int m, n, k;
    DistanceType type;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DistanceInputs<T>& dims) {
    return os;
}

template <typename T>
struct OutStruct {
    T* dist;
};

template <typename T>
struct InStruct {
};

template <typename T>
class DistanceTest: public ::testing::TestWithParam<DistanceInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<DistanceInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int m = params.m;
        int n = params.n;
        int k = params.k;
        allocate(x, m*k);
        allocate(y, n*k);
        allocate(out_ref, m*n);
        allocate(out, m*n);
        r.uniform(x, m*k, T(-1.0), T(1.0));
        r.uniform(y, n*k, T(-1.0), T(1.0));
        OutStruct<T> outval = { out };
        InStruct<T> inval;
        naiveDistance(out_ref, x, y, m, n, k, params.type);
        char* workspace = nullptr;
        size_t workspaceSize = 0;
        typedef cutlass::Shape<8, 128, 128> OutputTile_t;

        distance<T,T,OutStruct<T>,InStruct<T>,OutputTile_t>(outval, x, y, m, n, k,
                                                inval, params.type,
                                                (void*)workspace, workspaceSize);
        if (workspaceSize != 0) {
            allocate(workspace, workspaceSize);
        }
        distance<T,T,OutStruct<T>,InStruct<T>,OutputTile_t>(outval, x, y, m, n, k,
                                                inval, params.type,
                                                (void*)workspace, workspaceSize);
        CUDA_CHECK(cudaFree(workspace));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    DistanceInputs<T> params;
    T *x, *y, *out_ref, *out;
};

const std::vector<DistanceInputs<float> > inputsf = {
    {0.001f, 1024, 1024,   32, EucExpandedL2,       1234ULL}, // accumulate issue due to x^2 + y^2 -2xy
    {0.001f, 1024,   32, 1024, EucExpandedL2,       1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedL2,       1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedL2,       1234ULL},

    {0.001f, 1024, 1024,   32, EucExpandedL2Sqrt,   1234ULL},
    {0.001f, 1024,   32, 1024, EucExpandedL2Sqrt,   1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedL2Sqrt,   1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedL2Sqrt,   1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL2,     1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL2,     1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL2,     1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL2,     1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL2Sqrt, 1234ULL},

    {0.001f, 1024, 1024,   32, EucExpandedCosine,   1234ULL},
    {0.001f, 1024,   32, 1024, EucExpandedCosine,   1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedCosine,   1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedCosine,   1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL1,     1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL1,     1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL1,     1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL1,     1234ULL},
};

const std::vector<DistanceInputs<double> > inputsd = {
    {0.001f, 1024, 1024,   32, EucExpandedL2,       1234ULL}, // accumulate issue due to x^2 + y^2 -2xy
    {0.001f, 1024,   32, 1024, EucExpandedL2,       1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedL2,       1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedL2,       1234ULL},

    {0.001f, 1024, 1024,   32, EucExpandedL2Sqrt,   1234ULL},
    {0.001f, 1024,   32, 1024, EucExpandedL2Sqrt,   1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedL2Sqrt,   1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedL2Sqrt,   1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL2,     1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL2,     1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL2,     1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL2,     1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL2Sqrt, 1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL2Sqrt, 1234ULL},

    {0.001f, 1024, 1024,   32, EucExpandedCosine,   1234ULL},
    {0.001f, 1024,   32, 1024, EucExpandedCosine,   1234ULL},
    {0.001f,   32, 1024, 1024, EucExpandedCosine,   1234ULL},
    {0.001f, 1024, 1024, 1024, EucExpandedCosine,   1234ULL},

    {0.001f, 1024, 1024,   32, EucUnexpandedL1,     1234ULL},
    {0.001f, 1024,   32, 1024, EucUnexpandedL1,     1234ULL},
    {0.001f,   32, 1024, 1024, EucUnexpandedL1,     1234ULL},
    {0.001f, 1024, 1024, 1024, EucUnexpandedL1,     1234ULL},
};

///@todo: enable these tests, once 'distance' function is implemented
typedef DistanceTest<float> DistanceTestF;
TEST_P(DistanceTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.m, params.n,
                            CompareApprox<float>(params.tolerance)));
}

///@todo: enable these tests, once 'distance' function is implemented
typedef DistanceTest<double> DistanceTestD;
TEST_P(DistanceTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.m, params.n,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceTestD, ::testing::ValuesIn(inputsd));

} // end namespace Distance
} // end namespace MLCommon
