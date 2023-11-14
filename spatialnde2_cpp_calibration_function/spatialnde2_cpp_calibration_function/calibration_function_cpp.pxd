from libcpp.memory cimport shared_ptr

from spatialnde2.recmath cimport math_function


cdef extern from "calibration_function_cpp.hpp" namespace "snde2_fn_ex" nogil:
    cdef shared_ptr[math_function] define_calibration_function()
    cdef shared_ptr[math_function] calibration_function
    pass
