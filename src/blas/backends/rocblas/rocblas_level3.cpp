/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include "rocblas_helper.hpp"
#include "rocblas_task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/rocblas/onemkl_blas_rocblas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm(Func func, cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                 int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define GEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,    \
              int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                    \
              cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c,  \
              int64_t ldc) {                                                                       \
        gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
    }

GEMM_LAUNCHER(float, rocblas_sgemm)
GEMM_LAUNCHER(double, rocblas_dgemm)
GEMM_LAUNCHER(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<bfloat16, 1> &a, int64_t lda,
          cl::sycl::buffer<bfloat16, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    throw unimplemented("blas", "gemm", "for column_major layout");
}


template <typename Func, typename T_A, typename T_B, typename T_C, typename DATATYPE_A,
          typename DATATYPE_B, typename DATATYPE_C>
inline void gemm(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                 cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, T_C alpha, cl::sycl::buffer<T_A, 1> &a, int64_t lda,
                 cl::sycl::buffer<T_B, 1> &b, int64_t ldb, T_C beta, cl::sycl::buffer<T_C, 1> &c,
                 int64_t ldc) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    using cuDataType_C = typename CudaEquivalentType<T_C>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType_A *>(a_acc);
            auto b_ = sc.get_mem<cuDataType_B *>(b_acc);
            auto c_ = sc.get_mem<cuDataType_C *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType_C *)&alpha, a_,
                              DT_A, lda, b_, DT_B, ldb, (cuDataType_C *)&beta, c_, DT_C, ldc, DT_C,
                              rocblas_gemm_algo_standard);
        });
    });
}
#ifdef ENABLE_HALF_ROUTINES

#define GEMM_EX_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, \
                         CUDADATATYPE_C)                                                         \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
              int64_t k, TYPE_C alpha, cl::sycl::buffer<TYPE_A, 1> &a, int64_t lda,              \
              cl::sycl::buffer<TYPE_B, 1> &b, int64_t ldb, TYPE_C beta,                          \
              cl::sycl::buffer<TYPE_C, 1> &c, int64_t ldc) {                                     \
        gemm(ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, CUDADATATYPE_C, queue, transa,      \
             transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                              \
    }
#else
#define GEMM_EX_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, \
                         CUDADATATYPE_C)                                                         \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
              int64_t k, TYPE_C alpha, cl::sycl::buffer<TYPE_A, 1> &a, int64_t lda,              \
              cl::sycl::buffer<TYPE_B, 1> &b, int64_t ldb, TYPE_C beta,                          \
              cl::sycl::buffer<TYPE_C, 1> &c, int64_t ldc) {                                     \
              throw unimplemented("blas", "gemm", "half is disabled");                           \
    }
#endif
GEMM_EX_LAUNCHER(cl::sycl::half, cl::sycl::half, float, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_32F)
GEMM_EX_LAUNCHER(cl::sycl::half, cl::sycl::half, cl::sycl::half, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_16F)

#undef GEMM_EX_LAUNCHER

template <typename Func, typename T>
inline void symm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }

SYMM_LAUNCHER(float, rocblas_ssymm)
SYMM_LAUNCHER(double, rocblas_dsymm)
SYMM_LAUNCHER(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER

template <typename Func, typename T>
inline void hemm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define HEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }
HEMM_LAUNCHER(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER

template <typename Func, typename T>
inline void syrk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda, T beta,
                 cl::sycl::buffer<T, 1> &c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda,
                              (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYRK_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, TYPE beta,                \
              cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                                     \
        syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);    \
    }

SYRK_LAUNCHER(float, rocblas_ssyrk)
SYRK_LAUNCHER(double, rocblas_dsyrk)
SYRK_LAUNCHER(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void herk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, ScalarType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                 ScalarType beta, cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuScalarType *)&alpha, a_, lda,
                              (cuScalarType *)&beta, c_, ldc);
        });
    });
}

#define HERK_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                      \
    void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
              SCALAR_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda, SCALAR_TYPE beta, \
              cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                    \
        herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);        \
    }

HERK_LAUNCHER(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER

template <typename Func, typename T>
inline void syr2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                  cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                  int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
}

#define SYR2K_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                       \
    void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,    \
               TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                              \
               cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, \
               int64_t ldc) {                                                                      \
        syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,     \
              ldc);                                                                                \
    }
SYR2K_LAUNCHER(float, rocblas_ssyr2k)
SYR2K_LAUNCHER(double, rocblas_dsyr2k)
SYR2K_LAUNCHER(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void her2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, DataType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                  cl::sycl::buffer<DataType, 1> &b, int64_t ldb, ScalarType beta,
                  cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuScalarType *)&beta, c_, ldc);
        });
    });
}

#define HER2K_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                  \
    void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
               DATA_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda,                 \
               cl::sycl::buffer<DATA_TYPE, 1> &b, int64_t ldb, SCALAR_TYPE beta,                \
               cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                \
        her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,  \
              ldc);                                                                             \
    }

HER2K_LAUNCHER(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline void trmm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                              get_rocblas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb);
        });
    });
}

#define TRMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRMM_LAUNCHER(float, rocblas_strmm)
TRMM_LAUNCHER(double, rocblas_dtrmm)
TRMM_LAUNCHER(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER

template <typename Func, typename T>
inline void trsm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                              get_rocblas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb);
        });
    });
}

#define TRSM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRSM_LAUNCHER(float, rocblas_strsm)
TRSM_LAUNCHER(double, rocblas_dtrsm)
TRSM_LAUNCHER(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline cl::sycl::event gemm(Func func, cl::sycl::queue &queue, transpose transa, transpose transb,
                            int64_t m, int64_t n, int64_t k, T alpha, const T *a, int64_t lda,
                            const T *b, int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define GEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,  \
                         int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,           \
                         const TYPE *b, int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,            \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, \
                    c, ldc, dependencies);                                                       \
    }

GEMM_LAUNCHER_USM(float, rocblas_sgemm)
GEMM_LAUNCHER_USM(double, rocblas_dgemm)
GEMM_LAUNCHER_USM(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER_USM(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER_USM
template <typename Func, typename T_A, typename T_B, typename T_C, typename DATATYPE_A,
          typename DATATYPE_B, typename DATATYPE_C>
inline cl::sycl::event gemm(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                            cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                            int64_t n, int64_t k, T_C alpha, const T_A *a, int64_t lda,
                            const T_B *b, int64_t ldb, T_C beta, T_C *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    using cuDataType_C = typename CudaEquivalentType<T_C>::Type;
    overflow_check(m, n, k, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler &sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<const cuDataType_A *>(a);
            auto b_ = reinterpret_cast<const cuDataType_B *>(b);
            auto c_ = reinterpret_cast<cuDataType_C *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType_C *)&alpha, a_,
                              DT_A, lda, b_, DT_B, ldb, (cuDataType_C *)&beta, c_, DT_C, ldc, c_, DT_C,
                              ldc, DT_C, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_pack_int8x4);
        });
    });
    return done;
}

#define GEMM_EX_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, ROCBLAS_ROUTINE, CUDADATATYPE_A,               \
                             CUDADATATYPE_B, CUDADATATYPE_C)                                       \
    cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,    \
                         int64_t n, int64_t k, TYPE_C alpha, const TYPE_A *a, int64_t lda,         \
                         const TYPE_B *b, int64_t ldb, TYPE_C beta, TYPE_C *c, int64_t ldc,        \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return gemm(ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, CUDADATATYPE_C, queue, transa, \
                    transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);           \
    }

GEMM_EX_LAUNCHER_USM(half, half, float, rocblas_gemm_ex, rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f32_r)
GEMM_EX_LAUNCHER_USM(half, half, half, rocblas_gemm_ex, rocblas_datatype_f16_r, rocblas_datatype_f16_r, rocblas_datatype_f16_r)

#undef GEMM_EX_LAUNCHER_USM

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const bfloat16 *a, int64_t lda,
                     const bfloat16 *b, int64_t ldb, float beta, float *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", "for column_major layout");
}
template <typename Func, typename T>
inline cl::sycl::event symm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            int64_t m, int64_t n, T alpha, const T *a, int64_t lda, const T *b,
                            int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define SYMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,   \
                         int64_t n, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,       \
                         int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                 \
    }

SYMM_LAUNCHER_USM(float, rocblas_ssymm)
SYMM_LAUNCHER_USM(double, rocblas_dsymm)
SYMM_LAUNCHER_USM(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER_USM(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event hemm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            int64_t m, int64_t n, T alpha, const T *a, int64_t lda, const T *b,
                            int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), m, n, (cuDataType *)&alpha, a_,
                              lda, b_, ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define HEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,   \
                         int64_t n, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,       \
                         int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                 \
    }
HEMM_LAUNCHER_USM(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER_USM(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event syrk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            int64_t n, int64_t k, T alpha, const T *a, int64_t lda, T beta, T *c,
                            int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda,
                              (cuDataType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define SYRK_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                         int64_t k, TYPE alpha, const TYPE *a, int64_t lda, TYPE beta, TYPE *c,   \
                         int64_t ldc,                                                             \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                \
    }

SYRK_LAUNCHER_USM(float, rocblas_ssyrk)
SYRK_LAUNCHER_USM(double, rocblas_dsyrk)
SYRK_LAUNCHER_USM(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline cl::sycl::event herk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            int64_t n, int64_t k, const ScalarType alpha, const DataType *a,
                            int64_t lda, const ScalarType beta, DataType *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuScalarType *)&alpha, a_, lda,
                              (cuScalarType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define HERK_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                         int64_t k, const SCALAR_TYPE alpha, const DATA_TYPE *a, int64_t lda,     \
                         const SCALAR_TYPE beta, DATA_TYPE *c, int64_t ldc,                       \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                \
    }

HERK_LAUNCHER_USM(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER_USM(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event syr2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                             int64_t n, int64_t k, T alpha, const T *a, int64_t lda, const T *b,
                             int64_t ldb, T beta, T *c, int64_t ldc,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuDataType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define SYR2K_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                          int64_t k, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,        \
                          int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                            \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, \
                     c, ldc, dependencies);                                                        \
    }
SYR2K_LAUNCHER_USM(float, rocblas_ssyr2k)
SYR2K_LAUNCHER_USM(double, rocblas_dsyr2k)
SYR2K_LAUNCHER_USM(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline cl::sycl::event her2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                             int64_t n, int64_t k, const DataType alpha, const DataType *a,
                             int64_t lda, const DataType *b, int64_t ldb, const ScalarType beta,
                             DataType *c, int64_t ldc,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<DataType>::Type;
    using cuScalarType = typename CudaEquivalentType<ScalarType>::Type;
    overflow_check(n, k, lda, ldb, ldc);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_fill_mode(upper_lower),
                              get_rocblas_operation(trans), n, k, (cuDataType *)&alpha, a_, lda, b_,
                              ldb, (cuScalarType *)&beta, c_, ldc);
        });
    });
    return done;
}

#define HER2K_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                          int64_t k, const DATA_TYPE alpha, const DATA_TYPE *a, int64_t lda,       \
                          const DATA_TYPE *b, int64_t ldb, const SCALAR_TYPE beta, DATA_TYPE *c,   \
                          int64_t ldc,                                                             \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, \
                     c, ldc, dependencies);                                                        \
    }

HER2K_LAUNCHER_USM(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER_USM(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER_USM

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline cl::sycl::event trmm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                            const T *a, int64_t lda, T *b, int64_t ldb,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<cuDataType *>(b);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                              get_rocblas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb);
        });
    });
    return done;
}

#define TRMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                    \
    cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower,                \
                         transpose trans, diag unit_diag, int64_t m, int64_t n, TYPE alpha,        \
                         const TYPE *a, int64_t lda, TYPE *b, int64_t ldb,                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                    a, lda, b, ldb, dependencies);                                                 \
    }
TRMM_LAUNCHER_USM(float, rocblas_strmm)
TRMM_LAUNCHER_USM(double, rocblas_dtrmm)
TRMM_LAUNCHER_USM(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER_USM(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event trsm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                            const T *a, int64_t lda, T *b, int64_t ldb,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldb);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<cuDataType *>(b);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_side_mode(left_right),
                              get_rocblas_fill_mode(upper_lower), get_rocblas_operation(trans),
                              get_rocblas_diag_type(unit_diag), m, n, (cuDataType *)&alpha, a_, lda,
                              b_, ldb);
        });
    });
    return done;
}

#define TRSM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                    \
    cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower,                \
                         transpose trans, diag unit_diag, int64_t m, int64_t n, TYPE alpha,        \
                         const TYPE *a, int64_t lda, TYPE *b, int64_t ldb,                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                    a, lda, b, ldb, dependencies);                                                 \
    }
TRSM_LAUNCHER_USM(float, rocblas_strsm)
TRSM_LAUNCHER_USM(double, rocblas_dtrsm)
TRSM_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER_USM

} // namespace column_major
namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm(Func func, cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                 int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}

#define GEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,    \
              int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                    \
              cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c,  \
              int64_t ldc) {                                                                       \
        gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
    }

GEMM_LAUNCHER(float, rocblas_sgemm)
GEMM_LAUNCHER(double, rocblas_dgemm)
GEMM_LAUNCHER(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER

template <typename Func, typename T_A, typename T_B, typename T_C, typename DATATYPE_A,
          typename DATATYPE_B, typename DATATYPE_C>
inline void gemm(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                 cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, T_C alpha, cl::sycl::buffer<T_A, 1> &a, int64_t lda,
                 cl::sycl::buffer<T_B, 1> &b, int64_t ldb, T_C beta, cl::sycl::buffer<T_C, 1> &c,
                 int64_t ldc) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}

#define GEMM_EX_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, \
                         CUDADATATYPE_C)                                                         \
    void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,  \
              int64_t k, TYPE_C alpha, cl::sycl::buffer<TYPE_A, 1> &a, int64_t lda,              \
              cl::sycl::buffer<TYPE_B, 1> &b, int64_t ldb, TYPE_C beta,                          \
              cl::sycl::buffer<TYPE_C, 1> &c, int64_t ldc) {                                     \
        gemm(ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, CUDADATATYPE_C, queue, transa,      \
             transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                              \
    }
GEMM_EX_LAUNCHER(cl::sycl::half, cl::sycl::half, float, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_32F)
GEMM_EX_LAUNCHER(cl::sycl::half, cl::sycl::half, cl::sycl::half, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_16F)
#undef GEMM_EX_LAUNCHER

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<bfloat16, 1> &a, int64_t lda,
          cl::sycl::buffer<bfloat16, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}

template <typename Func, typename T>
inline void symm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    throw unimplemented("blas", "symm", "for row_major layout");
}

#define SYMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }

SYMM_LAUNCHER(float, rocblas_ssymm)
SYMM_LAUNCHER(double, rocblas_dsymm)
SYMM_LAUNCHER(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER

template <typename Func, typename T>
inline void hemm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                 int64_t n, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                 cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                 int64_t ldc) {
    throw unimplemented("blas", "hemm", "for row_major layout");
}

#define HEMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                        \
    void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,     \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, cl::sycl::buffer<TYPE, 1> &b, \
              int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                 \
        hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, \
             ldc);                                                                                 \
    }
HEMM_LAUNCHER(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER

template <typename Func, typename T>
inline void syrk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda, T beta,
                 cl::sycl::buffer<T, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "syrk", "for row_major layout");
}

#define SYRK_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
              TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda, TYPE beta,                \
              cl::sycl::buffer<TYPE, 1> &c, int64_t ldc) {                                     \
        syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);    \
    }

SYRK_LAUNCHER(float, rocblas_ssyrk)
SYRK_LAUNCHER(double, rocblas_dsyrk)
SYRK_LAUNCHER(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void herk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                 int64_t k, ScalarType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                 ScalarType beta, cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "herk", "for row_major layout");
}

#define HERK_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                      \
    void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,     \
              SCALAR_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda, SCALAR_TYPE beta, \
              cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                    \
        herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);        \
    }

HERK_LAUNCHER(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER

template <typename Func, typename T>
inline void syr2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, T alpha, cl::sycl::buffer<T, 1> &a, int64_t lda,
                  cl::sycl::buffer<T, 1> &b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> &c,
                  int64_t ldc) {
    throw unimplemented("blas", "syr2k", "for row_major layout");
}

#define SYR2K_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                       \
    void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,    \
               TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,                              \
               cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, TYPE beta, cl::sycl::buffer<TYPE, 1> &c, \
               int64_t ldc) {                                                                      \
        syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,     \
              ldc);                                                                                \
    }
SYR2K_LAUNCHER(float, rocblas_ssyr2k)
SYR2K_LAUNCHER(double, rocblas_dsyr2k)
SYR2K_LAUNCHER(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER

template <typename Func, typename DataType, typename ScalarType>
inline void her2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                  int64_t k, DataType alpha, cl::sycl::buffer<DataType, 1> &a, int64_t lda,
                  cl::sycl::buffer<DataType, 1> &b, int64_t ldb, ScalarType beta,
                  cl::sycl::buffer<DataType, 1> &c, int64_t ldc) {
    throw unimplemented("blas", "her2k", "for row_major layout");
}

#define HER2K_LAUNCHER(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                  \
    void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
               DATA_TYPE alpha, cl::sycl::buffer<DATA_TYPE, 1> &a, int64_t lda,                 \
               cl::sycl::buffer<DATA_TYPE, 1> &b, int64_t ldb, SCALAR_TYPE beta,                \
               cl::sycl::buffer<DATA_TYPE, 1> &c, int64_t ldc) {                                \
        her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,  \
              ldc);                                                                             \
    }

HER2K_LAUNCHER(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline void trmm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "trmm", "for row_major layout");
}

#define TRMM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRMM_LAUNCHER(float, rocblas_strmm)
TRMM_LAUNCHER(double, rocblas_dtrmm)
TRMM_LAUNCHER(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER

template <typename Func, typename T>
inline void trsm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                 transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                 cl::sycl::buffer<T, 1> &a, int64_t lda, cl::sycl::buffer<T, 1> &b, int64_t ldb) {
    throw unimplemented("blas", "trsm", "for row_major layout");
}

#define TRSM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                    \
    void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,      \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a,  \
              int64_t lda, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb) {                        \
        trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, \
             lda, b, ldb);                                                                     \
    }
TRSM_LAUNCHER(float, rocblas_strsm)
TRSM_LAUNCHER(double, rocblas_dtrsm)
TRSM_LAUNCHER(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER

// USM APIs

template <typename Func, typename T>
inline cl::sycl::event gemm(Func func, cl::sycl::queue &queue, transpose transa, transpose transb,
                            int64_t m, int64_t n, int64_t k, T alpha, const T *a, int64_t lda,
                            const T *b, int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}

#define GEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,  \
                         int64_t n, int64_t k, TYPE alpha, const TYPE *a, int64_t lda,           \
                         const TYPE *b, int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,            \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return gemm(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, \
                    c, ldc, dependencies);                                                       \
    }

GEMM_LAUNCHER_USM(float, rocblas_sgemm)
GEMM_LAUNCHER_USM(double, rocblas_dgemm)
GEMM_LAUNCHER_USM(std::complex<float>, rocblas_cgemm)
GEMM_LAUNCHER_USM(std::complex<double>, rocblas_zgemm)

#undef GEMM_LAUNCHER_USM
template <typename Func, typename T_A, typename T_B, typename T_C, typename DATATYPE_A,
          typename DATATYPE_B, typename DATATYPE_C>
inline cl::sycl::event gemm(Func func, DATATYPE_A DT_A, DATATYPE_B DT_B, DATATYPE_C DT_C,
                            cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                            int64_t n, int64_t k, T_C alpha, const T_A *a, int64_t lda,
                            const T_B *b, int64_t ldb, T_C beta, T_C *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}

#define GEMM_EX_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, ROCBLAS_ROUTINE, CUDADATATYPE_A,               \
                             CUDADATATYPE_B, CUDADATATYPE_C)                                       \
    cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,    \
                         int64_t n, int64_t k, TYPE_C alpha, const TYPE_A *a, int64_t lda,         \
                         const TYPE_B *b, int64_t ldb, TYPE_C beta, TYPE_C *c, int64_t ldc,        \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return gemm(ROCBLAS_ROUTINE, CUDADATATYPE_A, CUDADATATYPE_B, CUDADATATYPE_C, queue, transa, \
                    transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);           \
    }

GEMM_EX_LAUNCHER_USM(half, half, float, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_32F)
GEMM_EX_LAUNCHER_USM(half, half, half, rocblas_gemm_ex, HIP_R_16F, HIP_R_16F, HIP_R_16F)

#undef GEMM_EX_LAUNCHER_USM

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const bfloat16 *a, int64_t lda,
                     const bfloat16 *b, int64_t ldb, float beta, float *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", "for row_major layout");
}
template <typename Func, typename T>
inline cl::sycl::event symm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            int64_t m, int64_t n, T alpha, const T *a, int64_t lda, const T *b,
                            int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", "for row_major layout");
}

#define SYMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,   \
                         int64_t n, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,       \
                         int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return symm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                 \
    }

SYMM_LAUNCHER_USM(float, rocblas_ssymm)
SYMM_LAUNCHER_USM(double, rocblas_dsymm)
SYMM_LAUNCHER_USM(std::complex<float>, rocblas_csymm)
SYMM_LAUNCHER_USM(std::complex<double>, rocblas_zsymm)

#undef SYMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event hemm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            int64_t m, int64_t n, T alpha, const T *a, int64_t lda, const T *b,
                            int64_t ldb, T beta, T *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "hemm", "for row_major layout");
}

#define HEMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,   \
                         int64_t n, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,       \
                         int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {          \
        return hemm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, \
                    beta, c, ldc, dependencies);                                                 \
    }
HEMM_LAUNCHER_USM(std::complex<float>, rocblas_chemm)
HEMM_LAUNCHER_USM(std::complex<double>, rocblas_zhemm)

#undef HEMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event syrk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            int64_t n, int64_t k, T alpha, const T *a, int64_t lda, T beta, T *c,
                            int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", "for row_major layout");
}

#define SYRK_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                         int64_t k, TYPE alpha, const TYPE *a, int64_t lda, TYPE beta, TYPE *c,   \
                         int64_t ldc,                                                             \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return syrk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                \
    }

SYRK_LAUNCHER_USM(float, rocblas_ssyrk)
SYRK_LAUNCHER_USM(double, rocblas_dsyrk)
SYRK_LAUNCHER_USM(std::complex<float>, rocblas_csyrk)
SYRK_LAUNCHER_USM(std::complex<double>, rocblas_zsyrk)

#undef SYRK_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline cl::sycl::event herk(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            int64_t n, int64_t k, const ScalarType alpha, const DataType *a,
                            int64_t lda, const ScalarType beta, DataType *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "herk", "for row_major layout");
}

#define HERK_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                         int64_t k, const SCALAR_TYPE alpha, const DATA_TYPE *a, int64_t lda,     \
                         const SCALAR_TYPE beta, DATA_TYPE *c, int64_t ldc,                       \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return herk(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, \
                    dependencies);                                                                \
    }

HERK_LAUNCHER_USM(std::complex<float>, float, rocblas_cherk)
HERK_LAUNCHER_USM(std::complex<double>, double, rocblas_zherk)

#undef HERK_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event syr2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                             int64_t n, int64_t k, T alpha, const T *a, int64_t lda, const T *b,
                             int64_t ldb, T beta, T *c, int64_t ldc,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", "for row_major layout");
}

#define SYR2K_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                   \
    cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                          int64_t k, TYPE alpha, const TYPE *a, int64_t lda, const TYPE *b,        \
                          int64_t ldb, TYPE beta, TYPE *c, int64_t ldc,                            \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return syr2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, \
                     c, ldc, dependencies);                                                        \
    }
SYR2K_LAUNCHER_USM(float, rocblas_ssyr2k)
SYR2K_LAUNCHER_USM(double, rocblas_dsyr2k)
SYR2K_LAUNCHER_USM(std::complex<float>, rocblas_csyr2k)
SYR2K_LAUNCHER_USM(std::complex<double>, rocblas_zsyr2k)

#undef SYR2K_LAUNCHER_USM

template <typename Func, typename DataType, typename ScalarType>
inline cl::sycl::event her2k(Func func, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                             int64_t n, int64_t k, const DataType alpha, const DataType *a,
                             int64_t lda, const DataType *b, int64_t ldb, const ScalarType beta,
                             DataType *c, int64_t ldc,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "her2k", "for row_major layout");
}

#define HER2K_LAUNCHER_USM(DATA_TYPE, SCALAR_TYPE, ROCBLAS_ROUTINE)                                 \
    cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,    \
                          int64_t k, const DATA_TYPE alpha, const DATA_TYPE *a, int64_t lda,       \
                          const DATA_TYPE *b, int64_t ldb, const SCALAR_TYPE beta, DATA_TYPE *c,   \
                          int64_t ldc,                                                             \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return her2k(ROCBLAS_ROUTINE, queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, \
                     c, ldc, dependencies);                                                        \
    }

HER2K_LAUNCHER_USM(std::complex<float>, float, rocblas_cher2k)
HER2K_LAUNCHER_USM(std::complex<double>, double, rocblas_zher2k)

#undef HER2K_LAUNCHER_USM

// NOTE: In rocblas TRMM diverted from the netlib blas and for performance
// reason it requires the C matrix to be
// separated from the B matrix. It is possible to use B instead of C, but this
// will slow-down the code.
template <typename Func, typename T>
inline cl::sycl::event trmm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                            const T *a, int64_t lda, T *b, int64_t ldb,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", "for row_major layout");
}

#define TRMM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                    \
    cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower,                \
                         transpose trans, diag unit_diag, int64_t m, int64_t n, TYPE alpha,        \
                         const TYPE *a, int64_t lda, TYPE *b, int64_t ldb,                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return trmm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                    a, lda, b, ldb, dependencies);                                                 \
    }
TRMM_LAUNCHER_USM(float, rocblas_strmm)
TRMM_LAUNCHER_USM(double, rocblas_dtrmm)
TRMM_LAUNCHER_USM(std::complex<float>, rocblas_ctrmm)
TRMM_LAUNCHER_USM(std::complex<double>, rocblas_ztrmm)

#undef TRMM_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event trsm(Func func, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                            transpose trans, diag unit_diag, int64_t m, int64_t n, T alpha,
                            const T *a, int64_t lda, T *b, int64_t ldb,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", "for row_major layout");
}

#define TRSM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                    \
    cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower,                \
                         transpose trans, diag unit_diag, int64_t m, int64_t n, TYPE alpha,        \
                         const TYPE *a, int64_t lda, TYPE *b, int64_t ldb,                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {            \
        return trsm(ROCBLAS_ROUTINE, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, \
                    a, lda, b, ldb, dependencies);                                                 \
    }
TRSM_LAUNCHER_USM(float, rocblas_strsm)
TRSM_LAUNCHER_USM(double, rocblas_dtrsm)
TRSM_LAUNCHER_USM(std::complex<float>, rocblas_ctrsm)
TRSM_LAUNCHER_USM(std::complex<double>, rocblas_ztrsm)

#undef TRSM_LAUNCHER_USM

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi