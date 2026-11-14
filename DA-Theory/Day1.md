# Day 1: Python Data Science Prerequisites — NumPy & Pandas Fundamentals

---

## 1. Introduction to NumPy & Numerical Computing

### 1.1 What is NumPy?
**NumPy** (Numerical Python) is the foundational open-source Python library for scientific computing, data analytics, and machine learning. It provides:
- A high-performance multidimensional array object called **`ndarray`**.
- Tools for integrating C/C++ and Fortran code.
- Mathematical functions for fast operations on arrays without explicit Python loops (vectorization).
- Linear algebra, Fourier transform, and random number capabilities.

---

### 1.2 Memory Architecture: Python List vs. NumPy `ndarray`

A fundamental concept frequently tested in exams is why NumPy is drastically faster and more memory-efficient than native Python lists.

| Dimension | Native Python `list` | NumPy `ndarray` |
| :--- | :--- | :--- |
| **Memory Allocation** | **Non-contiguous (Scattered):** Stores an array of *pointers/references* to objects scattered across the heap memory. | **Contiguous Memory Buffer:** Stores raw binary values in a single continuous block of memory. |
| **Data Types** | **Heterogeneous:** Can store integers, strings, floats, and objects in the same list. | **Homogeneous:** Every element must have the exact same data type (`dtype`). |
| **Type Checking Overhead** | **Dynamic Type Checking:** Python checks the type of each element at runtime during every operation. | **Zero Overhead:** Explicit type allocation via `dtype` allows compiled C code to operate directly on raw memory. |
| **Cache Performance** | **Poor CPU Cache Locality:** Pointer chasing causes CPU cache misses. | **Optimal CPU Cache Locality:** Contiguous memory enables CPU pre-fetching and **SIMD** (Single Instruction Multiple Data) vectorization. |
| **Memory Footprint** | Large (header + pointer + PyObject wrapper for every element). | Extremely small (only raw bytes + small array metadata header). |

#### Code Illustration: Memory Footprint
```python
import sys
import numpy as np

size = 1000
python_list = list(range(size))
numpy_array = np.arange(size)

# Python list memory: size of pointer array + size of each integer PyObject
actual_list_size = sys.getsizeof(python_list) + sum(sys.getsizeof(item) for item in python_list)
numpy_size = numpy_array.nbytes  # Total bytes consumed by elements

print("Python List Memory:", actual_list_size, "bytes")  # ~28,000+ bytes
print("NumPy Array Memory:", numpy_size, "bytes")        # Exactly 4,000 or 8,000 bytes
```

---

### 1.3 List Comprehension vs. Generator Expression

Understanding iteration efficiency is critical for preprocessing large datasets:

- **List Comprehension `[x for x in data]`:** Eagerly computes all elements and holds the entire collection in memory at once. Fast for indexing, but can exhaust RAM for massive datasets.
- **Generator Expression `(x for x in data)`:** Evaluates elements lazily (on-demand) one by one using the iterator protocol (`next()` or `__next__()`). Consumes constant, negligible memory regardless of dataset size.

```python
import sys

val = 100_000
list_comp = [x for x in range(val)]
gen_exp = (x for x in range(val))

print("List Comprehension RAM:", sys.getsizeof(list_comp), "bytes")  # ~800 KB
print("Generator Expression RAM:", sys.getsizeof(gen_exp), "bytes")  # ~104 bytes (constant!)
```

---

## 2. NumPy Array Creation & Attributes

### 2.1 Factory Functions

NumPy provides specialized constructor functions to instantiate arrays efficiently without manually looping:

| Constructor | Syntax | Output Description | Exam & Practical Use Case |
| :--- | :--- | :--- | :--- |
| `np.array()` | `np.array([1, 2, 3], dtype=np.float32)` | Converts Python sequence into an `ndarray`. | Initializing fixed data with explicit precision. |
| `np.zeros()` | `np.zeros((rows, cols))` | Allocates an array filled with zeros (`0.0`). | Pre-allocating output memory buffers. |
| `np.ones()` | `np.ones((rows, cols))` | Allocates an array filled with ones (`1.0`). | Initializing bias weights or baseline masks. |
| `np.full()` | `np.full((2, 3), 7)` | Allocates an array filled with a constant value. | Initializing constant matrices (e.g., prior baselines). |
| `np.eye()` | `np.eye(N)` | Generates an $N \times N$ 2D identity matrix. | Linear algebra, diagonal scaling, identity transformations. |
| `np.empty()` | `np.empty((2, 2))` | Allocates uninitialized memory buffer (contains garbage values). | Maximum allocation speed when values will be immediately overwritten. |
| `np.arange()` | `np.arange(start, stop, step)` | Generates evenly spaced values within half-open interval $[start, stop)$. | Index arrays, discrete step iteration. |
| `np.linspace()`| `np.linspace(start, stop, num)`| Generates `num` evenly spaced points over closed interval $[start, stop]$. | Continuous mathematical function evaluation. |
| `default_rng()`| `np.random.default_rng(seed=42)`| Modern NumPy random number generator instance. | Reproducible stochastic simulation, model weight initialization. |

### 2.2 Key `ndarray` Metadata Attributes

Every NumPy array exposes key metadata properties stored in its C header:
- **`arr.ndim`**: Number of array dimensions (axes). Vector = 1, Matrix = 2, Tensor = 3+.
- **`arr.shape`**: A tuple of integers showing the size of the array along each dimension.
- **`arr.size`**: Total number of elements across all dimensions (product of `shape` components).
- **`arr.dtype`**: The data type of elements (e.g., `int32`, `int64`, `float32`, `float64`).
- **`arr.itemsize`**: Number of bytes occupied by a single array element (e.g., 8 bytes for `float64`).
- **`arr.nbytes`**: Total bytes consumed by array data: $\text{nbytes} = \text{size} \times \text{itemsize}$.

---

## 3. Vectorization & Broadcasting

### 3.1 What is Vectorization?
**Vectorization** is the execution of operations on entire arrays at once rather than looping over individual elements in Python. 
- Python `for` loops execute in interpreted bytecode, checking types on every iteration.
- Vectorized operations run compiled C loops at hardware execution speeds, taking advantage of CPU vector registers (**SIMD**).

```python
inch_measures = np.array([1.0, 2.5, 5.0, 10.0])
cms_measures = inch_measures * 2.54  # Vectorized: multiplied in C without a Python loop!
```

### 3.2 The Rules of Broadcasting
**Broadcasting** describes how NumPy treats arrays with different shapes during arithmetic operations without creating unnecessary memory copies.

The smaller array is "broadcast" across the larger array so that they have compatible shapes.

#### The 2 Formal Broadcasting Rules:
1. **Rule 1 (Dimension Alignment):** If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its **leading (left) side**.
2. **Rule 2 (Compatibility Check):** If the shape of the two arrays does not match in any dimension, the array with shape equal to `1` in that dimension is stretched to match the other shape.
3. **Incompatible Shapes:** If in any dimension the sizes disagree and neither equals `1`, a `ValueError: operands could not be broadcast together` is raised.

#### Example Walkthrough:
- Array $A$: Shape `(4, 3)`
- Array $B$: Shape `(3,)`
- Step 1: Pad $B$ on left $\to$ Shape `(1, 3)`
- Step 2: Compare dimensions from right to left:
  - Dim 2: $3$ and $3 \implies$ Match
  - Dim 1: $4$ and $1 \implies$ Compatible ($1$ stretches to $4$)
- Result: Operation succeeds with output shape `(4, 3)`.

### 3.3 Boolean Masking & Vectorized Filtering
Boolean indexing creates an array of truth values used to filter elements efficiently:

```python
rng = np.random.default_rng(seed=42)
signal = rng.uniform(0.0, 1.0, size=(4, 4))

# 1. Generate boolean mask
mask = signal > 0.7   # Returns 2D boolean array

# 2. Apply mask (returns 1D array of matching values)
high_signals = signal[mask]
```

---

## 4. Indexing, Slicing & The "View vs. Copy" Rule

### 4.1 Basic Slicing Syntax
Syntax: `arr[start:stop:step]`
- `start` is inclusive.
- `stop` is exclusive.
- Negative step `[::-1]` reverses the array.
- Negative indices count backwards from the end (`-1` is the last element).

```python
arr = np.arange(10, 101, 10)  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(arr[2:5])               # [30, 40, 50]
print(arr[1:8:2])             # [20, 40, 60, 80]
print(arr[::-1])              # Reversed array
```

### 4.2 CRITICAL EXAM CONCEPT: Views vs. Copies

This is one of the most common MCQ trap areas in Data Analytics exams.

| Mechanism | Basic Slicing (`arr[2:5]`) | Fancy / Boolean Indexing (`arr[[1, 3]]` or `arr[mask]`) |
| :--- | :--- | :--- |
| **Return Type** | **View** | **Copy** |
| **Underlying Memory** | **Shares the exact same memory buffer** as the original array. No new data is allocated. | **Allocates a brand new memory buffer** on the heap. |
| **Mutation Side Effect** | **Modifying the slice modifies the original array!** | Modifying the indexed copy **does NOT** change the original array. |
| **Performance** | Extremely fast, zero memory overhead. | Slower, requires memory allocation and copying. |

```python
# View Mutation Demonstration
original = np.array([10, 20, 30, 40, 50])
slice_view = original[1:4]
slice_view[0] = 999  # Mutating view

print(original)  # Output: [ 10, 999,  30,  40,  50] -> ORIGINAL WAS MUTATED!

# Explicit Copy Creation
safe_copy = original[1:4].copy()
safe_copy[0] = 111
print(original)  # Original remains unchanged: [ 10, 999,  30,  40,  50]
```

---

## 5. Universal Functions (`ufuncs`) & Mathematical Operations

Universal functions (`ufuncs`) perform fast, element-by-element operations on ndarrays:

- **Arithmetic Ufuncs:** `np.add(x, y)`, `np.subtract(x, y)`, `np.multiply(x, y)`, `np.divide(x, y)`, `np.power(x, y)`.
- **Rounding Ufuncs:**
  - `np.floor()`: Rounds down to the nearest lower integer ($\lfloor 2.9 \rfloor = 2.0$, $\lfloor -2.1 \rfloor = -3.0$).
  - `np.ceil()`: Rounds up to the nearest higher integer ($\lceil 2.1 \rceil = 3.0$, $\lceil -2.9 \rceil = -2.0$).
  - `np.round()`: Rounds to nearest even integer (banker's rounding in Python 3).
- **Statistical Aggregations:** `np.sum()`, `np.mean()`, `np.std()`, `np.var()`, `np.min()`, `np.max()`, `np.argmin()`, `np.argmax()`, `np.cumsum()`.

---

## 6. Linear Algebra (`np.linalg`)

Linear algebra operations form the mathematical foundation of data analytics, PCA, and machine learning models.

### 6.1 Multiplication Types: Element-wise vs. Dot Product

| Operation | Operator / Function | Mathematical Meaning | Shape Compatibility Rule |
| :--- | :--- | :--- | :--- |
| **Element-wise Product** | `*` or `np.multiply(A, B)` | Multiplies matching coordinates: $C_{ij} = A_{ij} \times B_{ij}$. | Shapes must be identical or broadcastable. |
| **Vector Dot Product** | `np.dot(u, v)` or `u @ v` | Sum of products: $\sum u_i v_i$. | Vectors must have identical length. |
| **Matrix Multiplication** | `@` (Python 3.5+) or `np.matmul(A, B)` | Standard row-column inner products. | Inner dimensions must match: $(m \times k) @ (k \times n) \to (m \times n)$. |

### 6.2 The 1D Transpose MCQ Trap
In NumPy, transposing a 1D array does **NOT** turn it into a 2D column vector:
```python
v = np.array([1, 2, 3])
print(v.shape)    # (3,)
print(v.T.shape)  # Still (3,)! Transposing a 1D array has NO effect!

# To convert to a column vector (shape (3, 1)):
v_col = v[:, np.newaxis]   # shape (3, 1)
# Or:
v_col = v.reshape(-1, 1)   # shape (3, 1)
```

### 6.3 Core Linear Algebra Functions
- `A.T`: Transpose matrix.
- `np.trace(A)`: Sum of main diagonal elements of matrix $A$.
- `np.linalg.inv(A)`: Computes matrix inverse $A^{-1}$ (requires $A$ to be square and non-singular, $\det(A) \neq 0$).
- `np.linalg.det(A)`: Computes determinant of square matrix.
- `np.linalg.eig(A)`: Computes eigenvalues and right eigenvectors.
- `np.linalg.solve(A, b)`: Solves linear equation system $Ax = b$ efficiently without explicit inversion.

---

## 7. Advanced Tensors & Einstein Summation (`np.einsum`)

Einstein summation (`np.einsum`) provides a concise, high-speed syntax for evaluating multidimensional tensor products and dimension transformations.

### 7.1 Common Einsum Expressions
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Matrix Multiplication (ij, jk -> ik)
matmul = np.einsum('ij,jk->ik', A, B)

# 2. Matrix Transpose (ij -> ji)
transpose = np.einsum('ij->ji', A)

# 3. Matrix Trace (ii -> scalar)
trace = np.einsum('ii->', A)  # Sums diagonal elements
```

### 7.2 Deep Learning Tensor Reshaping: PyTorch to TensorFlow
Image tensor conventions differ across deep learning frameworks:
- **PyTorch Convention:** `NCHW` $\to$ `(Batch, Channels, Height, Width)`
- **TensorFlow Convention:** `NHWC` $\to$ `(Batch, Height, Width, Channels)`

```python
tensor_nchw = np.zeros((2, 3, 4, 4))  # 2 images, 3 color channels, 4x4 pixels

# Conversion via einsum:
tensor_nhwc = np.einsum('nchw->nhwc', tensor_nchw)

# Equivalent conversion via np.transpose:
tensor_nhwc_alt = np.transpose(tensor_nchw, (0, 2, 3, 1))
print(tensor_nhwc.shape)  # Output: (2, 4, 4, 3)
```

---

## 8. PyTorch Framework Architecture Overview

PyTorch is a premier open-source deep learning framework based on Torch, created by Meta AI and maintained under the Linux Foundation. Built with a Python interface and an optimized C++ backend, it accelerates multidimensional tensor computation on GPUs.

### Core PyTorch Modules & APIs

| Module | Core Purpose | Common Classes / APIs |
| :--- | :--- | :--- |
| **`torch.nn`** | Neural network layers, containers, activation functions, and loss functions. | `nn.Module`, `nn.Linear`, `nn.Conv2d`, `nn.CrossEntropyLoss`, `nn.MSELoss` |
| **`torch.optim`** | Optimization algorithms for updating neural network weights via gradients. | `optim.SGD`, `optim.Adam`, `optim.AdamW`, `optim.RMSprop` |
| **`torch.utils.data`** | Data ingestion pipelines, custom dataset wrapping, multiprocessing batch loading. | `Dataset`, `DataLoader`, `random_split` |
| **`torch.autograd`** | Automatic differentiation engine calculating vector-Jacobian products for backpropagation. | `Tensor.backward()`, `torch.no_grad()`, `torch.enable_grad()` |
| **`torch.distributed`** | Multi-GPU and multi-node distributed training orchestration. | `DDP` (Distributed Data Parallel), `FSDP` (Fully Sharded Data Parallel) |

---

## 9. Introduction to Pandas: Series Fundamentals

### 9.1 What is Pandas?
**Pandas** is Python's primary data manipulation and analytics library. Built on top of NumPy, it adds labeled rows and columns, structured schemas, handling of missing data, and relational operations.

### 9.2 The Pandas Series
A **Series** is a 1-dimensional labeled array capable of holding any data type (`int`, `float`, `string`, `object`, `bool`).
- Consists of two components: the **index labels** and the underlying **NumPy array of values**.

```python
import pandas as pd
import numpy as np

# Creating Series from list
players = pd.Series(["sachin", "rohit", "virat"], index=[10, 20, 30], name="Cricketers")

# Creating Series from dictionary
roles = pd.Series({"sachin": "Batsman", "zaheer": "Bowler", "dhoni": "Keeper"})
```

### 9.3 Index Alignment with Missing Keys
When creating a Series from a dictionary and providing an explicit index, Pandas aligns keys to index labels. Keys not found in the dictionary receive `NaN` (floating-point missing value):

```python
data = {'a': 0.0, 'b': 1.0, 'c': 2.0}
s = pd.Series(data, index=['b', 'c', 'd', 'a'])
# Output:
# b    1.0
# c    2.0
# d    NaN  <- Missing key automatically converted to NaN
# a    0.0
```

### 9.4 Series Metadata Attributes
- `s.values`: Returns the underlying data buffer as a NumPy array.
- `s.index`: Returns the Index object containing labels.
- `s.dtype`: Data type of elements.
- `s.ndim`: Always `1` for any Series.
- `s.shape`: Tuple indicating length `(n,)`.
- `s.size`: Total element count.
- `s.nbytes`: Raw bytes of value buffer.

### 9.5 Memory Optimization: `object` vs. `category`
Strings stored as generic Python `object` dtypes waste substantial memory due to Python string pointer overhead.
- Converting repetitive string columns to **`category`** dtype replaces repeated strings with integer keys mapped to a small lookup dictionary.
- Memory usage is typically reduced by **70% to 90%**!

```python
raw_labels = np.random.choice(["Low", "Medium", "High"], size=10_000)
s_obj = pd.Series(raw_labels)
s_cat = s_obj.astype("category")

print("Object RAM:", s_obj.memory_usage(deep=True))    # ~600 KB
print("Category RAM:", s_cat.memory_usage(deep=True))  # ~10 KB (98% reduction!)
```

---

## 10. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **NumPy Array Homogeneity:** Every element in a NumPy array must have the identical `dtype`. If mixed types are supplied, NumPy implicitly upcasts them (e.g., `[1, 2.5, "hi"]` becomes all strings `U32`).
2. **Slicing is a View:** Basic slice notation `arr[a:b]` creates a view. Mutating the slice modifies the original array!
3. **Fancy Indexing is a Copy:** `arr[[0, 2]]` or `arr[arr > 5]` allocates and returns a completely new copy.
4. **1D Transpose Trap:** For a 1D vector `v` of shape `(N,)`, `v.T` has shape `(N,)`. Transposition only flips axes for 2D+ arrays.
5. **Broadcasting Dimension Rule:** Trailing dimensions are matched first from right to left. Leading dimensions can be padded with 1.
6. **Matrix Multiplication Operator:** `@` was introduced in Python 3.5. `*` is strictly element-wise multiplication.
7. **PyTorch autograd role:** `torch.autograd` performs automatic gradient computation; `torch.nn` defines network architectures.
8. **Pandas Series ndim:** A Pandas Series always has `ndim = 1`. A DataFrame has `ndim = 2`.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| `np.dot(A, B)` | `A * B` | `A * B` is element-wise multiplication; `np.dot(A, B)` or `A @ B` is row-column matrix multiplication. |
| `arr[1:3]` (Slice) | `arr[[1, 3]]` (Fancy index) | Slice returns a **view** (mutates original). Fancy index returns a **copy** (safe from mutations). |
| `np.arange` | `np.linspace` | `arange` takes `step` size and excludes stop; `linspace` takes total `num` of points and includes stop. |
| `s.memory_usage()` | `s.memory_usage(deep=True)` | Default `memory_usage()` only measures pointer references for objects; `deep=True` measures true memory. |
| `torch.nn` | `torch.optim` | `torch.nn` holds model layers and loss functions; `torch.optim` contains parameter update algorithms (SGD, Adam). |
| `np.floor` | `np.trunc` | For positive numbers both round down. For negative numbers, `floor(-2.1) = -3.0`, whereas `trunc(-2.1) = -2.0`. |

