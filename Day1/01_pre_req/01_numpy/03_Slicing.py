'''
numpy slicing is a powerful feature that allows you to extract specific portions of an array. 
Advantages of numpy slicing include:
1. Zero-copy views: sharing underlying buffer memory with the original array, which saves memory and improves performance.
2. Flexible indexing: Returns a view of the original array
3. Boolean indexing: Allows you to select elements based on conditions, making it easy to filter data.
'''

import numpy as np

sensor_reading = np.arange(9)
print("Original Array:", sensor_reading)

sensor_reading[3] = 999
print("Original Array:", sensor_reading)

sensor_reading[3] = 3
print("Sliced Array:", sensor_reading[2:5])  # Slicing from index 2 to 4 (5 is exclusive)

# tensor change
sensor_reading[2:5] = 100
print("Modified Array:", sensor_reading)  # The original array is modified

print("_" * 50)
values = np.arange(10,101,10)
print("Original Array:", values)
print("values[2]:", values[2])  # Accessing a single element
print("values[2:]", values[2:])  # Slicing from index 2 to the end  
print("values[2:5]:", values[2:5])  # Slicing
print("values[:5]:", values[:5])  # Slicing from the start to index 4
print("values[1:8:2]:", values[1:8:2])  # Slicing with a step of 2
print("values[::3]:", values[::3])  # Slicing with a step of 3
print("values[-1]:", values[-1])  # Accessing the last element
print("values[-2]:", values[-2])  # Accessing the penultimate element
print("values[-3]:", values[-3])  # Accessing the third-to-last element
print("values[-3:]:", values[-3:]) # Slicing the last three elements
print("values[:-3]:", values[:-3]) # Slicing all but the last three elements
print("values[-5:-2]:", values[-5:-2]) # Slicing from the fifth-to-last to the third-to-last element
print("values[-2:-5]:", values[-2:-5]) # This will return an empty array
print("values[-2:-5:-1]:", values[-2:-5:-1]) # Slicing in reverse order from the second-to-last to the fifth-to-last element
print("values[-5:-2:-1]:", values[-5:-2:-1]) # This will return an empty array
print("values[-2::-1]:", values[-2::-1]) # Slicing in reverse order from the second-to-last to the start of the array
print("values[::-1]:", values[::-1]) # Slicing in reverse order for the entire array
print("_" * 50)

feature_matrix = np.array([
    [1, 2, 3, 4], 
    [ 5, 6, 7, 8], 
    [7, 8, 9, 10], 
    [11, 12, 13, 14]
    ])

print("Original Array:\n", feature_matrix)
print("feature_matrix[1]:", feature_matrix[1])  # Accessing the second row
print("feature_matrix[1, 2]:", feature_matrix[1, 2])
print("feature_matrix[:1]:", feature_matrix[:1])  # Slicing the first row
print("feature_matrix[:2]:\n", feature_matrix[:2])  # Slicing the first two rows
print("_" * 50)
print("feature_matrix[1:3]:\n", feature_matrix[1:3])  # Slicing rows 1 and 2
print("_" * 50)
print("Original Array:\n", feature_matrix)
print("feature_matrix[:, 2]:", feature_matrix[:, 2])  # Accessing the third column
print("feature_matrix[:, 1:3]:\n", feature_matrix[:, 1:3])  # Slicing columns 1 and 2
print("feature_matrix[1:3, 1:3]:\n", feature_matrix[1:3, 1:3])  # Slicing a submatrix
print("_" * 50)


matrix = np.array([
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12]
])
row_rank1 = matrix[1, :]  # 1D array
row_rank2 = matrix[1:2, :]  # 2D array
# string interpolation to print the shape of the arrays
print(f"Row with rank 1: {row_rank1}, shape: {row_rank1.shape}")
print(f"Row with rank 2:\n{row_rank2}\nshape: {row_rank2.shape}")

# Column Extractions
col_rank1 = matrix[:, 0]  # 1D array
col_rank2 = matrix[:, 0:1]  # 2D array
print(f"Column with rank 1: {col_rank1}, shape: {col_rank1.shape}")
print(f"Column with rank 2:\n{col_rank2}\nshape: {col_rank2.shape}")
print("_" * 50)
data_grid = np.array([
    [ 1,  2,  3],
    [ 4,  5,  6],
    [ 7,  8,  9],
    [10, 11, 12]
])
print("Original Array:\n", data_grid)
row_indices = np.arange(4)  # [0, 1, 2, 3]
col_indices = np.array([0, 2, 0, 1])  # Column indices as target
extracted_elements = data_grid[row_indices, col_indices]
print("Extracted Elements:", extracted_elements)  
print("_" * 50)
print("_" * 50)