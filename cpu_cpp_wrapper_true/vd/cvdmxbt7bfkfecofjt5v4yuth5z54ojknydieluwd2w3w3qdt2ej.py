"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu

async_compile.wait(globals())
del async_compile
"""


import torch
from torch._inductor.codecache import CppWrapperCodeCache

cpp_wrapper_src = (
'''
#include <pybind11/pybind11.h>
namespace py = pybind11;

class RAIIPyObject {
public:
    RAIIPyObject() : obj_(nullptr) {}
    RAIIPyObject(PyObject* obj) : obj_(obj) {}
    ~RAIIPyObject() {
        Py_XDECREF(obj_);
    }
    RAIIPyObject& operator=(const RAIIPyObject& other) {
        if (this != &other) {
            Py_XDECREF(obj_);
            obj_ = other.obj_;
            Py_XINCREF(obj_);
        }
        return *this;
    }
    operator PyObject*() {
        return obj_;
    }
    PyObject* get() {
        return obj_;
    }
private:
    PyObject* obj_;
};

#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
using namespace torch::aot_inductor;

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>

#include <c10/util/generic_math.h>
typedef at::Half half;
typedef at::BFloat16 bfloat16;

// Round up to the nearest multiple of 64
[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
// _tensor_constant0 device(type='cpu') torch.int64 (19,) (1,) 71a540460d70

#include "/home/lijinpei/development/try_inductor_cpp_wrapper/cpu_cpp_wrapper_true/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void cpp_fused_add_lift_fresh_0(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(19L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L)))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp2 = in_ptr1[static_cast<int64_t>(0L)];
                    auto tmp1 = at::vec::convert<float,1,int64_t,2>(tmp0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L) && x0 < static_cast<int64_t>(19L)))
                {
                    for (int64_t x0_tail = static_cast<int64_t>(16L);x0_tail < static_cast<int64_t>(19L); x0_tail++)
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(x0_tail)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(0L)];
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        out_ptr0[static_cast<int64_t>(x0_tail)] = tmp3;
                    }
                }
            }
        }
    }
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cpu);

void inductor_entry_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles  // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed)
) {
    py::gil_scoped_release release;

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
    auto arg0_1 = std::move(inputs[0]);
    [[maybe_unused]] auto _tensor_constant0 = std::move(inputs[1]);
    static constexpr int64_t int_array_0[] = {19L, };
    static constexpr int64_t int_array_1[] = {1L, };
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cpu,  0, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    cpp_fused_add_lift_fresh_0((const int64_t*)(_tensor_constant0.data_ptr()), (const float*)(arg0_1.data_ptr()), (float*)(buf0.data_ptr()));
    arg0_1.reset();
    output_handles[0] = buf0.release();
} // inductor_entry_impl

'''
)

inductor_entry = CppWrapperCodeCache.load_pybinding(
    ["std::vector<AtenTensorHandle>"], cpp_wrapper_src, "cpu", 1)

def _wrap_func(f):
    def g(args):
        input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
        constants_tensor = [_tensor_constant0]
        input_tensors.extend(constants_tensor)

        input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)

        args.clear()

        output_handles = f(input_handles)
        output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
        return output_tensors

    return g

call = _wrap_func(inductor_entry)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _tensor_constant0
    _tensor_constant0 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.int64)
    arg0_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
