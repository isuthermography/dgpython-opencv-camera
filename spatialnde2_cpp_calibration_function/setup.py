import sys
import os
import os.path
import numpy as np
from setuptools import setup,Extension
from Cython.Build import cythonize
import spatialnde2 as snde

def fixup_cmake_libs(libs):
    ret_libs = []
    ret_explicit = []
    for lib in libs:
        if lib.startswith("-l"):
            ret_libs.append(lib[2:])
            pass
        elif "/" in lib or "\\" in lib:
            ret_explicit.append(lib)
            pass
        else:
            ret_libs.append(lib)
            pass
        pass
    return (ret_explicit,ret_libs)

spatialnde2_compile_definitions = open(os.path.join(os.path.dirname(snde.__file__),"compile_definitions.txt")).read().strip().split(" ")
spatialnde2_compile_include_dirs = open(os.path.join(os.path.dirname(snde.__file__),"compile_include_dirs.txt")).read().strip().split(" ")
spatialnde2_compile_library_dirs = open(os.path.join(os.path.dirname(snde.__file__),"compile_library_dirs.txt")).read().strip().split(" ")
spatialnde2_compile_libraries_cmake = open(os.path.join(os.path.dirname(snde.__file__),"compile_libraries.txt")).read().strip().split(" ")

spatialnde2_compile_libraries_cmake.append('opencv_core460')
spatialnde2_compile_libraries_cmake.append('opencv_calib3d460')

while spatialnde2_compile_libraries_cmake.count("hdf5-shared") > 0:
    spatialnde2_compile_libraries_cmake[spatialnde2_compile_libraries_cmake.index("hdf5-shared")] = "hdf5"

while spatialnde2_compile_libraries_cmake.count("hdf5_cpp-shared") > 0:
    spatialnde2_compile_libraries_cmake[spatialnde2_compile_libraries_cmake.index("hdf5_cpp-shared")] = "hdf5_cpp"

(spatialnde2_compile_explicit_libraries,spatialnde2_compile_libraries) = fixup_cmake_libs(spatialnde2_compile_libraries_cmake)


ext_modules=cythonize(
    Extension("spatialnde2_cpp_calibration_function.calibration_function",
              sources=["spatialnde2_cpp_calibration_function/calibration_function.pyx" ],
              include_dirs=[os.path.dirname(snde.__file__)] + spatialnde2_compile_include_dirs,
              
              library_dirs=[os.path.dirname(snde.__file__)] + spatialnde2_compile_library_dirs,
              extra_compile_args = ["/Od", '/Zi', '/GS'] + spatialnde2_compile_definitions,
              extra_link_args = ['-debug'] + spatialnde2_compile_explicit_libraries,
              libraries=["spatialnde2"]  + spatialnde2_compile_libraries,
              undef_macros = ["NDEBUG"]
              ))

setup(name="spatialnde2_cpp_calibration_function",
            description="Calibration Function for dgpython-opencv-camera",
            author="Tyler Lesthaeghe",
            url="http://udri.udayton.edu",
            ext_modules=ext_modules,
            packages=["spatialnde2_cpp_calibration_function"])
