
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

#define CUDA_DRIVER_CHECK(EXPR)                    do {                                                   CUresult code = EXPR;                              const char *msg;                                   CUresult code_get_error = cuGetErrorString(code, &msg);     if (code_get_error != CUDA_SUCCESS) {                  throw std::runtime_error(                              std::string("CUDA driver error: ") +               std::string("invalid error code!"));       }                                                  if (code != CUDA_SUCCESS) {                            throw std::runtime_error(                              std::string("CUDA driver error: ") +               std::string(msg));                         }                                              } while (0);

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
