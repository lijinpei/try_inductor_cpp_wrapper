# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
_tensor_constant0 = None  # device(type='cpu') torch.int64 (19,) (1,) 769f54d65220


cpp_fused_add_lift_fresh_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'float*'], '''
#include "/home/lijinpei/development/try_inductor_cpp_wrapper/cpu_cpp_wrapper_false/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const int64_t* in_ptr0,
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
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    buf0 = empty_strided_cpu((19, ), (1, ), torch.float32)
    cpp_fused_add_lift_fresh_0(_tensor_constant0, arg0_1, buf0)
    del arg0_1
    return (buf0, )


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
