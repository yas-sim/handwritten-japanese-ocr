#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

std::vector<cv::RotatedRect> postProcess(
    const float *link_data_pointer, const std::vector<int> &link_shape, float link_conf_threshold,
    const float *cls_data_pointer,  const std::vector<int> &cls_shape,  float cls_conf_threshold,
    const int input_w, const int input_h);


extern "C" {

// Bridge function to the actual postprocess function
static PyObject* postprocess_bridge(PyObject* self, PyObject* args) {
    PyArrayObject *link_logits, *segm_logits;
    int input_w, input_h;
    float cls_conf_threshold, link_conf_threshold;

    // Parse parameters
    if(!PyArg_ParseTuple(args, 
        "O!O!iiff",                                         // link_logits, segm_logits, input_w, input_h, link_conf, cls_conf
        &PyArray_Type, &link_logits,
        &PyArray_Type, &segm_logits,
        &input_w, &input_h,                                 // Input image width and height
        &link_conf_threshold, &cls_conf_threshold)) {       // Argument is 2 Numpy objects
        return nullptr;
    }

    std::vector<int> link_shape;
    size_t l_ndims = PyArray_NDIM(link_logits);                               // Number of dimensions
    npy_intp* l_shape = PyArray_SHAPE(link_logits);                           // Shape
    for(size_t i=0; i<l_ndims; i++) {
        link_shape.push_back(l_shape[i]);
    }
    float *link_data_pointer = static_cast<float*>(PyArray_DATA(link_logits));

    std::vector<int> cls_shape;
    size_t s_ndims = PyArray_NDIM(segm_logits);                               // Number of dimensions
    npy_intp* s_shape = PyArray_SHAPE(segm_logits);                           // Shape
    for(size_t i=0; i<s_ndims; i++) {
        cls_shape.push_back(s_shape[i]);
    }
    float *cls_data_pointer = static_cast<float*>(PyArray_DATA(segm_logits));

    auto rects = postProcess(link_data_pointer, link_shape, link_conf_threshold,
                             cls_data_pointer,  cls_shape,  cls_conf_threshold, 
                             input_w, input_h);

    int out_size = rects.size();

    // Create a Numpy object to store result
    PyObject *output;
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
    std::vector<npy_intp> output_shape {out_size, 2+2+1 };       // Shape ( center(2)+size(2)+angle(1) )
    output = PyArray_Zeros(output_shape.size(), output_shape.data(), descr, 0);
    float* output_buf = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)));                            // Obtain pointer to the data

    for(size_t i=0; i<out_size; i++) {
        auto rect = rects[i];
        output_buf[ i * 5 + 0 ] = rect.center.x;
        output_buf[ i * 5 + 1 ] = rect.center.y;
        output_buf[ i * 5 + 2 ] = rect.size.width;
        output_buf[ i * 5 + 3 ] = rect.size.height;
        output_buf[ i * 5 + 4 ] = rect.angle;
    }
    return output;
}

// Function definition table to export to Python
PyMethodDef method_table[] = {
    {"postprocess", static_cast<PyCFunction>(postprocess_bridge), METH_VARARGS, "C++ version of postprocess for text detection model (text-detection-0003)"},
    {NULL, NULL, 0, NULL}
};

// Module definition table
PyModuleDef text_detection_postprocess_module = {
    PyModuleDef_HEAD_INIT,
    "text_detection_postprocess",     // m_name: Module Name
    "C++ version of text_detection postprocess. Supports text-detection-0003 model of OpenVINO OMZ",      // m_doc : Docstring for the module
    -1,
    method_table
};

// Initialize and register module function
// Function name must be 'PyInit_'+module name
// This function must be the only *non-static* function in the source code
PyMODINIT_FUNC PyInit_text_detection_postprocess(void) {
    import_array();                                 // Required to receive Numpy object as arguments
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return PyModule_Create(&text_detection_postprocess_module);
}

} // extern "C"
