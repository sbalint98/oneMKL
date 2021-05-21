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
#include "blas/function_table.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

#define WRAPPER_VERSION 1

extern "C" blas_function_table_t mkl_blas_table = {
    WRAPPER_VERSION,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dotc,
    oneapi::mkl::blas::cublas::column_major::dotc,
    oneapi::mkl::blas::cublas::column_major::dotu,
    oneapi::mkl::blas::cublas::column_major::dotu,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotm,
    oneapi::mkl::blas::cublas::column_major::rotm,
    oneapi::mkl::blas::cublas::column_major::rotmg,
    oneapi::mkl::blas::cublas::column_major::rotmg,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::sdsdot,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::ger,
    oneapi::mkl::blas::cublas::column_major::ger,
    oneapi::mkl::blas::cublas::column_major::gerc,
    oneapi::mkl::blas::cublas::column_major::gerc,
    oneapi::mkl::blas::cublas::column_major::geru,
    oneapi::mkl::blas::cublas::column_major::geru,
    oneapi::mkl::blas::cublas::column_major::hbmv,
    oneapi::mkl::blas::cublas::column_major::hbmv,
    oneapi::mkl::blas::cublas::column_major::hemv,
    oneapi::mkl::blas::cublas::column_major::hemv,
    oneapi::mkl::blas::cublas::column_major::her,
    oneapi::mkl::blas::cublas::column_major::her,
    oneapi::mkl::blas::cublas::column_major::her2,
    oneapi::mkl::blas::cublas::column_major::her2,
    oneapi::mkl::blas::cublas::column_major::hpmv,
    oneapi::mkl::blas::cublas::column_major::hpmv,
    oneapi::mkl::blas::cublas::column_major::hpr,
    oneapi::mkl::blas::cublas::column_major::hpr,
    oneapi::mkl::blas::cublas::column_major::hpr2,
    oneapi::mkl::blas::cublas::column_major::hpr2,
    oneapi::mkl::blas::cublas::column_major::sbmv,
    oneapi::mkl::blas::cublas::column_major::sbmv,
    oneapi::mkl::blas::cublas::column_major::spmv,
    oneapi::mkl::blas::cublas::column_major::spmv,
    oneapi::mkl::blas::cublas::column_major::spr,
    oneapi::mkl::blas::cublas::column_major::spr,
    oneapi::mkl::blas::cublas::column_major::spr2,
    oneapi::mkl::blas::cublas::column_major::spr2,
    oneapi::mkl::blas::cublas::column_major::symv,
    oneapi::mkl::blas::cublas::column_major::symv,
    oneapi::mkl::blas::cublas::column_major::syr,
    oneapi::mkl::blas::cublas::column_major::syr,
    oneapi::mkl::blas::cublas::column_major::syr2,
    oneapi::mkl::blas::cublas::column_major::syr2,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
#ifndef DISABLE_HALF_RUTINES
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
#endif
    oneapi::mkl::blas::cublas::column_major::hemm,
    oneapi::mkl::blas::cublas::column_major::hemm,
    oneapi::mkl::blas::cublas::column_major::herk,
    oneapi::mkl::blas::cublas::column_major::herk,
    oneapi::mkl::blas::cublas::column_major::her2k,
    oneapi::mkl::blas::cublas::column_major::her2k,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::trsm_batch,
    oneapi::mkl::blas::cublas::column_major::trsm_batch,
    oneapi::mkl::blas::cublas::column_major::trsm_batch,
    oneapi::mkl::blas::cublas::column_major::trsm_batch,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemm_bias,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::asum,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy,
    oneapi::mkl::blas::cublas::column_major::axpy_batch,
    oneapi::mkl::blas::cublas::column_major::axpy_batch,
    oneapi::mkl::blas::cublas::column_major::axpy_batch,
    oneapi::mkl::blas::cublas::column_major::axpy_batch,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::copy,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dot,
    oneapi::mkl::blas::cublas::column_major::dotc,
    oneapi::mkl::blas::cublas::column_major::dotc,
    oneapi::mkl::blas::cublas::column_major::dotu,
    oneapi::mkl::blas::cublas::column_major::dotu,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamin,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::iamax,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::nrm2,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rot,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotg,
    oneapi::mkl::blas::cublas::column_major::rotm,
    oneapi::mkl::blas::cublas::column_major::rotm,
    oneapi::mkl::blas::cublas::column_major::rotmg,
    oneapi::mkl::blas::cublas::column_major::rotmg,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::scal,
    oneapi::mkl::blas::cublas::column_major::sdsdot,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::swap,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gbmv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::gemv,
    oneapi::mkl::blas::cublas::column_major::ger,
    oneapi::mkl::blas::cublas::column_major::ger,
    oneapi::mkl::blas::cublas::column_major::gerc,
    oneapi::mkl::blas::cublas::column_major::gerc,
    oneapi::mkl::blas::cublas::column_major::geru,
    oneapi::mkl::blas::cublas::column_major::geru,
    oneapi::mkl::blas::cublas::column_major::hbmv,
    oneapi::mkl::blas::cublas::column_major::hbmv,
    oneapi::mkl::blas::cublas::column_major::hemv,
    oneapi::mkl::blas::cublas::column_major::hemv,
    oneapi::mkl::blas::cublas::column_major::her,
    oneapi::mkl::blas::cublas::column_major::her,
    oneapi::mkl::blas::cublas::column_major::her2,
    oneapi::mkl::blas::cublas::column_major::her2,
    oneapi::mkl::blas::cublas::column_major::hpmv,
    oneapi::mkl::blas::cublas::column_major::hpmv,
    oneapi::mkl::blas::cublas::column_major::hpr,
    oneapi::mkl::blas::cublas::column_major::hpr,
    oneapi::mkl::blas::cublas::column_major::hpr2,
    oneapi::mkl::blas::cublas::column_major::hpr2,
    oneapi::mkl::blas::cublas::column_major::sbmv,
    oneapi::mkl::blas::cublas::column_major::sbmv,
    oneapi::mkl::blas::cublas::column_major::spmv,
    oneapi::mkl::blas::cublas::column_major::spmv,
    oneapi::mkl::blas::cublas::column_major::spr,
    oneapi::mkl::blas::cublas::column_major::spr,
    oneapi::mkl::blas::cublas::column_major::spr2,
    oneapi::mkl::blas::cublas::column_major::spr2,
    oneapi::mkl::blas::cublas::column_major::symv,
    oneapi::mkl::blas::cublas::column_major::symv,
    oneapi::mkl::blas::cublas::column_major::syr,
    oneapi::mkl::blas::cublas::column_major::syr,
    oneapi::mkl::blas::cublas::column_major::syr2,
    oneapi::mkl::blas::cublas::column_major::syr2,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbmv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tbsv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpmv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::tpsv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trmv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::trsv,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::gemm,
    oneapi::mkl::blas::cublas::column_major::hemm,
    oneapi::mkl::blas::cublas::column_major::hemm,
    oneapi::mkl::blas::cublas::column_major::herk,
    oneapi::mkl::blas::cublas::column_major::herk,
    oneapi::mkl::blas::cublas::column_major::her2k,
    oneapi::mkl::blas::cublas::column_major::her2k,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::symm,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syrk,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::syr2k,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trmm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::trsm,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemm_batch,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::column_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dotc,
    oneapi::mkl::blas::cublas::row_major::dotc,
    oneapi::mkl::blas::cublas::row_major::dotu,
    oneapi::mkl::blas::cublas::row_major::dotu,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotm,
    oneapi::mkl::blas::cublas::row_major::rotm,
    oneapi::mkl::blas::cublas::row_major::rotmg,
    oneapi::mkl::blas::cublas::row_major::rotmg,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::sdsdot,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::ger,
    oneapi::mkl::blas::cublas::row_major::ger,
    oneapi::mkl::blas::cublas::row_major::gerc,
    oneapi::mkl::blas::cublas::row_major::gerc,
    oneapi::mkl::blas::cublas::row_major::geru,
    oneapi::mkl::blas::cublas::row_major::geru,
    oneapi::mkl::blas::cublas::row_major::hbmv,
    oneapi::mkl::blas::cublas::row_major::hbmv,
    oneapi::mkl::blas::cublas::row_major::hemv,
    oneapi::mkl::blas::cublas::row_major::hemv,
    oneapi::mkl::blas::cublas::row_major::her,
    oneapi::mkl::blas::cublas::row_major::her,
    oneapi::mkl::blas::cublas::row_major::her2,
    oneapi::mkl::blas::cublas::row_major::her2,
    oneapi::mkl::blas::cublas::row_major::hpmv,
    oneapi::mkl::blas::cublas::row_major::hpmv,
    oneapi::mkl::blas::cublas::row_major::hpr,
    oneapi::mkl::blas::cublas::row_major::hpr,
    oneapi::mkl::blas::cublas::row_major::hpr2,
    oneapi::mkl::blas::cublas::row_major::hpr2,
    oneapi::mkl::blas::cublas::row_major::sbmv,
    oneapi::mkl::blas::cublas::row_major::sbmv,
    oneapi::mkl::blas::cublas::row_major::spmv,
    oneapi::mkl::blas::cublas::row_major::spmv,
    oneapi::mkl::blas::cublas::row_major::spr,
    oneapi::mkl::blas::cublas::row_major::spr,
    oneapi::mkl::blas::cublas::row_major::spr2,
    oneapi::mkl::blas::cublas::row_major::spr2,
    oneapi::mkl::blas::cublas::row_major::symv,
    oneapi::mkl::blas::cublas::row_major::symv,
    oneapi::mkl::blas::cublas::row_major::syr,
    oneapi::mkl::blas::cublas::row_major::syr,
    oneapi::mkl::blas::cublas::row_major::syr2,
    oneapi::mkl::blas::cublas::row_major::syr2,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
#ifndef DISABLE_HALF_RUTINES
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
#endif
    oneapi::mkl::blas::cublas::row_major::hemm,
    oneapi::mkl::blas::cublas::row_major::hemm,
    oneapi::mkl::blas::cublas::row_major::herk,
    oneapi::mkl::blas::cublas::row_major::herk,
    oneapi::mkl::blas::cublas::row_major::her2k,
    oneapi::mkl::blas::cublas::row_major::her2k,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::trsm_batch,
    oneapi::mkl::blas::cublas::row_major::trsm_batch,
    oneapi::mkl::blas::cublas::row_major::trsm_batch,
    oneapi::mkl::blas::cublas::row_major::trsm_batch,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemm_bias,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::asum,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy,
    oneapi::mkl::blas::cublas::row_major::axpy_batch,
    oneapi::mkl::blas::cublas::row_major::axpy_batch,
    oneapi::mkl::blas::cublas::row_major::axpy_batch,
    oneapi::mkl::blas::cublas::row_major::axpy_batch,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::copy,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dot,
    oneapi::mkl::blas::cublas::row_major::dotc,
    oneapi::mkl::blas::cublas::row_major::dotc,
    oneapi::mkl::blas::cublas::row_major::dotu,
    oneapi::mkl::blas::cublas::row_major::dotu,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamin,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::iamax,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::nrm2,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rot,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotg,
    oneapi::mkl::blas::cublas::row_major::rotm,
    oneapi::mkl::blas::cublas::row_major::rotm,
    oneapi::mkl::blas::cublas::row_major::rotmg,
    oneapi::mkl::blas::cublas::row_major::rotmg,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::scal,
    oneapi::mkl::blas::cublas::row_major::sdsdot,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::swap,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gbmv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::gemv,
    oneapi::mkl::blas::cublas::row_major::ger,
    oneapi::mkl::blas::cublas::row_major::ger,
    oneapi::mkl::blas::cublas::row_major::gerc,
    oneapi::mkl::blas::cublas::row_major::gerc,
    oneapi::mkl::blas::cublas::row_major::geru,
    oneapi::mkl::blas::cublas::row_major::geru,
    oneapi::mkl::blas::cublas::row_major::hbmv,
    oneapi::mkl::blas::cublas::row_major::hbmv,
    oneapi::mkl::blas::cublas::row_major::hemv,
    oneapi::mkl::blas::cublas::row_major::hemv,
    oneapi::mkl::blas::cublas::row_major::her,
    oneapi::mkl::blas::cublas::row_major::her,
    oneapi::mkl::blas::cublas::row_major::her2,
    oneapi::mkl::blas::cublas::row_major::her2,
    oneapi::mkl::blas::cublas::row_major::hpmv,
    oneapi::mkl::blas::cublas::row_major::hpmv,
    oneapi::mkl::blas::cublas::row_major::hpr,
    oneapi::mkl::blas::cublas::row_major::hpr,
    oneapi::mkl::blas::cublas::row_major::hpr2,
    oneapi::mkl::blas::cublas::row_major::hpr2,
    oneapi::mkl::blas::cublas::row_major::sbmv,
    oneapi::mkl::blas::cublas::row_major::sbmv,
    oneapi::mkl::blas::cublas::row_major::spmv,
    oneapi::mkl::blas::cublas::row_major::spmv,
    oneapi::mkl::blas::cublas::row_major::spr,
    oneapi::mkl::blas::cublas::row_major::spr,
    oneapi::mkl::blas::cublas::row_major::spr2,
    oneapi::mkl::blas::cublas::row_major::spr2,
    oneapi::mkl::blas::cublas::row_major::symv,
    oneapi::mkl::blas::cublas::row_major::symv,
    oneapi::mkl::blas::cublas::row_major::syr,
    oneapi::mkl::blas::cublas::row_major::syr,
    oneapi::mkl::blas::cublas::row_major::syr2,
    oneapi::mkl::blas::cublas::row_major::syr2,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbmv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tbsv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpmv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::tpsv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trmv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::trsv,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::gemm,
    oneapi::mkl::blas::cublas::row_major::hemm,
    oneapi::mkl::blas::cublas::row_major::hemm,
    oneapi::mkl::blas::cublas::row_major::herk,
    oneapi::mkl::blas::cublas::row_major::herk,
    oneapi::mkl::blas::cublas::row_major::her2k,
    oneapi::mkl::blas::cublas::row_major::her2k,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::symm,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syrk,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::syr2k,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trmm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::trsm,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemm_batch,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
    oneapi::mkl::blas::cublas::row_major::gemmt,
};
