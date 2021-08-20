#include "transform_preds.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_transform_dets_forward", &affine_transform_dets_forward, "affine_transform_dets_forward");
  m.def("get_affine_transform_forward", &get_affine_transform_forward, "get_affine_transform_forward");
}
