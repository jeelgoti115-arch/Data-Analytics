import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

mat_mul = np.einsum('ij,jk->ik', A, B)
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix Multiplication (Einsum):\n", mat_mul)
print("_" * 50)


transpose_einsum = np.einsum('ij->ji', A)
print("\nEinsum Transpose:\n", transpose_einsum)

trace_einsum = np.einsum('ii->', A)
print("\nEinsum Trace:", trace_einsum)
print("Validate Trace with np.trace:", np.trace(A))

print("=" * 50)
# nd Array to simulate pytorch tensor operations (Batch, Channels, Height, Width)
tensor_A = np.zeros((2, 3, 4, 4))  # Batch of 2, 3 channels, 4x4 images
print("Tensor A Shape:", tensor_A.shape)
print("Tensor A:\n", tensor_A)
print("=" * 50)
# convert Axes from pytorch (NCHW) to TensorFlow (NHWC) format using einsum
tf_tensor = np.einsum('nchw->nhwc', tensor_A)
tf_tensor_np=np.transpose(tensor_A, (0, 2, 3, 1))  # Using np.transpose for comparison
print("tf_tensor shape is ", tf_tensor.shape)
print("tf_tensor_np shape is ", tf_tensor_np.shape)
# print("Tensor A in TensorFlow format (NHWC):\n", tf_tensor)
print("=" * 50)