#include "cublas_scope_handle_hipsycl.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

cublas_handle::~cublas_handle() noexcept(false) {
    for (auto &handle_pair : cublas_handle_mapper_) {
        cublasStatus_t err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                CUBLAS_ERROR_FUNC(cublasDestroy, err, handle);
                handle = nullptr;
            }
            delete handle_pair.second;
            handle_pair.second = nullptr;
        }
    }
    cublas_handle_mapper_.clear();
}

thread_local cublas_handle CublasScopedContextHandler::handle_helper = cublas_handle{};

CublasScopedContextHandler::CublasScopedContextHandler(cl::sycl::queue queue, cl::sycl::interop_handle& ih): interop_h(ih){}

cublasHandle_t CublasScopedContextHandler::get_handle(const cl::sycl::queue &queue){
    cl::sycl::device device = queue.get_device();
    int current_device = interop_h.get_native_device<cl::sycl::backend::cuda>();
    auto it = handle_helper.cublas_handle_mapper_.find(current_device);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        auto handle = it->second->load();
        return handle;
    }
    cublasHandle_t handle;
    cublasStatus_t err;
    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    auto insert_iter = handle_helper.cublas_handle_mapper_.insert(
        std::make_pair(current_device, new std::atomic<cublasHandle_t>(handle)));
    return handle;
}
} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi