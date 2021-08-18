#include "transform_preds.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transform_preds_forward", &transform_preds_forward, "transform_preds_forward");
}
