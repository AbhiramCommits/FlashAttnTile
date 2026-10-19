// pybind11 bindings for the optional CUDA C++ paged-decode baseline.

#include <torch/extension.h>

torch::Tensor paged_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor seq_lens,
    int64_t page_size,
    double softmax_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_decode", &paged_decode_cuda,
        "Paged decode attention (CUDA C++ baseline)",
        pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"),
        pybind11::arg("page_table"), pybind11::arg("seq_lens"),
        pybind11::arg("page_size"), pybind11::arg("softmax_scale") = -1.0);
}
