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


triton_poi_fused_add_lift_fresh_0 = async_compile.triton('triton_poi_fused_add_lift_fresh_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_lift_fresh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A23547187B4F9FF1D7D7F36075D265A6C492A923E50CCCBBD740FA7549FFBD55', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_lift_fresh_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    torch.cuda.set_device(0)
    stream0 = get_raw_stream(0)
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    stream0 = get_raw_stream(0)
    _tensor_constant0 = generate_example_value((19,), (1,), 'cuda:0', torch.int64, 0)
    arg0_1 = generate_example_value((1,), (1,), 'cuda:0', torch.float32, 0)
    buf0 = generate_example_value((19,), (1,), 'cuda:0', torch.float32, 0)
    triton_poi_fused_add_lift_fresh_0.run(_tensor_constant0, arg0_1, buf0, 19, grid=grid(19), stream=stream0)
    del _tensor_constant0, arg0_1, buf0

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
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

#include <c10/util/generic_math.h>
typedef at::Half half;
typedef at::BFloat16 bfloat16;

// Round up to the nearest multiple of 64
[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
#include <filesystem>
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>

#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    CUresult code_get_error = cuGetErrorString(code, &msg); \
    if (code_get_error != CUDA_SUCCESS) {          \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string("invalid error code!"));   \
    }                                              \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

namespace {

struct Grid {
    Grid(uint32_t x, uint32_t y, uint32_t z)
      : grid_x(x), grid_y(y), grid_z(z) {}
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;

    bool is_non_zero() {
        return grid_x > 0 && grid_y > 0 && grid_z > 0;
    }
};

}  // anonymous namespace

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
// _tensor_constant0 device(type='cuda', index=0) torch.int64 (19,) (1,) 7ae20dc9d090
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cuda);

static CUfunction triton_poi_fused_add_lift_fresh_0 = nullptr;


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

    AOTICudaGuard device_guard(0);
    static constexpr int64_t int_array_0[] = {19L, };
    static constexpr int64_t int_array_1[] = {1L, };
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda,  0, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [tensor, add], Original ATen: [aten.lift_fresh, aten.add]
    cudaStream_t stream0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(0, (void**)&stream0));
    if (triton_poi_fused_add_lift_fresh_0 == nullptr) {
        triton_poi_fused_add_lift_fresh_0 = loadKernel("/home/lijinpei/development/try_inductor_cpp_wrapper/gpu_cpp_wrapper_true/uk/cukdypqgimqkeikmbvstud5isjhwnqlxtgl3otoarpacqzpbmciw.cubin", "triton_poi_fused_add_lift_fresh_0", 0);
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(_tensor_constant0.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(arg0_1.data_ptr());
    CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(buf0.data_ptr());
    int var_3 = 19L;
    void* kernel_args_var_0[] = {&var_0, &var_1, &var_2, &var_3};
    Grid triton_poi_fused_add_lift_fresh_0_grid_0 = Grid(1L, 1L, 1L);
    if (triton_poi_fused_add_lift_fresh_0_grid_0.is_non_zero()) {
        launchKernel(triton_poi_fused_add_lift_fresh_0, triton_poi_fused_add_lift_fresh_0_grid_0.grid_x, triton_poi_fused_add_lift_fresh_0_grid_0.grid_y, triton_poi_fused_add_lift_fresh_0_grid_0.grid_z, 1, 0, kernel_args_var_0, stream0);
    }
    arg0_1.reset();
    output_handles[0] = buf0.release();
} // inductor_entry_impl

'''
)

inductor_entry = CppWrapperCodeCache.load_pybinding(
    ["std::vector<AtenTensorHandle>"], cpp_wrapper_src, "cuda", 1)

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
    _tensor_constant0 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
