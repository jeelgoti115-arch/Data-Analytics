'''
What is numpy?
Numpy is a Python library that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. It is widely used in scientific computing, data analysis, and machine learning due to its efficiency and ease of use.


Memory Model : Continuous memory buffers enable vectorization and zero-copy slicing.
Type Precision : Explicit allocation via `dtype` avoids dynamic type-checking overhead.
Structured Schemas : Heterogeneous record types facilitate low-level C-struct binary parsing. 
'''

# create a virtual environment
# python -m venv cdacvenv
# activate the virtual environment : ./cdacvenv/Scripts/activate.ps1
# pip install numpy


import numpy as np
import sys

print("Sys version:", sys.version)
print("NumPy version:", np.__version__)
print("Python version info:", sys.version_info)
print("_" * 60)
vector_1d = np.array([10, 20, 30, 40, 50])
matrix_2d = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])

print("1D Vector:", vector_1d)
print("2D Matrix:\n", matrix_2d)
print("_" * 60)

# Memory footprint comparison
size = 1000
python_list = list(range(size))
numpy_array = np.arange(size)
actual_size = sys.getsizeof(python_list) + sum(sys.getsizeof(item) for item in python_list)
print("Python list size:", actual_size)
print("NumPy array size:", numpy_array.nbytes)
print("_" * 60)


# list comprehension vs generator expression
# val = 100000
# list_comp = [val for val in range(val)]
# print("List comprehension size:", sys.getsizeof(list_comp))
# gen_exp = (val for val in range(val))
# print("Generator expression size:", sys.getsizeof(gen_exp))

# print("next(gen_exp) 1:", next(gen_exp))
# print("next(gen_exp) 2:", next(gen_exp))
# print("next(gen_exp) 3:", next(gen_exp))
# print("gen_exp.__next__ 4:", gen_exp.__next__())
# print("gen_exp.__next__ 5:", gen_exp.__next__())
# print("gen_exp.__next__ 6:", gen_exp.__next__())

# numpy tensor initialization
# tensor factory functions
zero_matrix = np.zeros((3, 4))
one_matrix = np.ones((2, 3))
constant_matrix = np.full((2, 3), 7)
ident_matrix = np.eye(3)

random_rng = np.random.default_rng(seed=42)
random_matrix = random_rng.random((2, 3))

print("Zero Matrix:\n", zero_matrix)
print("One Matrix:\n", one_matrix)
print("Constant Matrix:\n", constant_matrix)
print("Identity Matrix:\n", ident_matrix)
print("Random Matrix:\n", random_matrix)