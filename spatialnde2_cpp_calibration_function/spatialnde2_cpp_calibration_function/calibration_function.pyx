# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uintptr_t

import spatialnde2 as snde



from calibration_function_cpp cimport calibration_function as calibration_function_cpp

# scalar_multiply_function_cpp is a shared_ptr to an
# snde::math_function To make it accessible to general python, we need
# to get a SWIG-wrapped pointer. This SWIG-wrapped version will be
# externally accessible, facilitated by
# the import in __init__.py,

# Note that for this to be reliably accessible scalar_multiply_function_cpp
# must be assigned in the .hpp file and therefore defined in this module
# and not linked in (otherwise we run afowl of the c++ static initialization
# ordering issues!)

calibration_function = snde.math_function.from_raw_shared_ptr(<uintptr_t>&calibration_function_cpp)



### As an alternative to defining/registering the math function
### in C++ code, we could alternatively do it here in Cython.
### This commented code below illustrates how:

## Create the Cython shared pointer
#cdef shared_ptr[math_function] snde2_example_scalar_multiply_function = define_scalar_multiply()


## Create the swig-wrapped object by wrapping the Cython shared pointer
## As a Python object this is externally accessible, facilitated by
## the import in __init__.py, as:
##   spatialnde2_example_cpp_function.scalar_multiply_function()
#scalar_multiply_function = snde.math_function.from_raw_shared_ptr(<uintptr_t>&snde2_example_scalar_multiply_function)

## Register the math function into the C++ database
## This should use the same python-accessible name for
## maximum interoperability

#snde.register_math_function("spatialnde2_example_cpp_function.scalar_multiply_function",scalar_multiply_function) 

