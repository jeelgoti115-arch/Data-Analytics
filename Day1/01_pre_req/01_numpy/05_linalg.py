import numpy as np

vector_a = np.array([9, 10])
vector_b = np.array([11, 12])

print("Vector A:", vector_a)
print("Vector B:", vector_b)
print("\n=== Dot Product ===")
print("Operator (np.dot):", np.dot(vector_a, vector_b)) # 9 * 11 + 10 * 12 = 219
print("Ufunc (np.inner):", np.inner(vector_a, vector_b)) # 9 * 11 + 10 * 12 = 219
# operator @ is used for matrix multiplication in Python 3.5 and later
print("Operator (@):", vector_a @ vector_b) # 9 * 11 + 10 * 12 = 219

print("_" * 50)
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print("Matrix X:\n", x)
print("Matrix Y:\n", y)
print("\nElement-wise Product (X * Y):\n", x * y)

print("\n=== Matrix Multiplication ===")
print("Operator (np.dot):\n", np.dot(x, y))
print("Operator (@):\n", x @ y)

print("_" * 50)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix:\n", matrix)
print("Transpose of Matrix:\n", matrix.T)  # Transpose of the matrix
print("_" * 50)

v_1d = np.array([1, 2, 3])
print("1D Vector:", v_1d)
print("Shape of 1D Vector:", v_1d.shape, "Transposed: v_1d", v_1d.T.shape)  # Transpose of a 1D vector does not change its shape
print("_" * 50)

scalar_bias = 5

matrix = matrix + scalar_bias  # Broadcasting: Adding a scalar to a matrix
print("Matrix after adding scalar bias:\n", matrix) 
print("_" * 50)
matrix_data = np.array([
    [ 1,  2,  3],
    [ 4,  5,  6],
    [ 7,  8,  9]
])

row_vector = np.array([10, 20, 30])
row_broadcasted = matrix_data + row_vector
print("Original Matrix:\n", matrix_data)
print("Row Vector:\n", row_vector)
print("Row Vector Broadcast Result:\n", row_broadcasted)

col_vector = np.array([[100], [200], [300]])
col_broadcasted = matrix_data + col_vector
print("Column Vector:\n", col_vector)
print("Column Vector Broadcast Result:\n", col_broadcasted)
print("_" * 50)
print("_" * 50)