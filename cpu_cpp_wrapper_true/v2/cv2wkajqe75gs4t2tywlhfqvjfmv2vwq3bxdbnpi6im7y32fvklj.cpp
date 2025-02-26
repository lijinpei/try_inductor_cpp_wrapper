
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


// Python bindings to call inductor_entry_cpp():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}


#include <torch/csrc/inductor/aoti_torch/c/shim.h>

static inline std::vector<AtenTensorHandle> unpack_tensor_handle_list(PyObject* pyvec) {
    std::vector<AtenTensorHandle> result;
    size_t result_len = PyList_GET_SIZE(pyvec);
    result.reserve(result_len);
    for (size_t i = 0; i < result_len; i++) {
        // AtenTensorHandle is essentially a pointer
        void* elem = PyCapsule_GetPointer(PyList_GET_ITEM(pyvec, i), NULL);
        result.push_back(reinterpret_cast<AtenTensorHandle>(elem));
    }
    return result;
}

static inline PyObject* pack_tensor_handle_list(const std::vector<AtenTensorHandle>& cppvec) {
    size_t result_len = cppvec.size();
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(result_len));
    for (size_t i = 0; i < result_len; i++) {
        PyObject *elem =
            cppvec[i] == nullptr
                ? Py_None
                // Store AtenTensorHandle as PyCapsulate
                : PyCapsule_New(reinterpret_cast<void*>(cppvec[i]), NULL, NULL);
        PyList_SET_ITEM(result, i, elem);
    }
    return result;
}

template <> inline std::vector<AtenTensorHandle> parse_arg<std::vector<AtenTensorHandle>>(PyObject* args, size_t n) {
    return unpack_tensor_handle_list(PyTuple_GET_ITEM(args, n));
}

PyObject* inductor_entry_cpp(std::vector<AtenTensorHandle>&& input_handles) {
    // For outputs, we only allocate a vector to hold returned tensor handles,
    // not allocating the actual output tensor storage here
    std::vector<AtenTensorHandle> output_handles(1);
    try {
        inductor_entry_impl(input_handles.data(), output_handles.data());
        if (PyErr_Occurred()) {
            return nullptr;
        }
        return pack_tensor_handle_list(output_handles);
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}


static PyObject* inductor_entry_cpp_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 1))
            throw std::runtime_error("requires 1 args");
        return inductor_entry_cpp(parse_arg<std::vector<AtenTensorHandle>>(args, 0));
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"inductor_entry_cpp", inductor_entry_cpp_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "inductor_entry_cpp", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_inductor_entry_cpp(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
