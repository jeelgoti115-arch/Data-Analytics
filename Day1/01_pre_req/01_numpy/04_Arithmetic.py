import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

print("Matrix X:\n", x)
print("Matrix Y:\n", y)

print("\n=== Addition ===")
print("Operator (+):\n", x + y)
print("Ufunc (np.add):\n", np.add(x, y))

print("\n=== Element-wise Multiplication ===")
print("Operator (*):\n", x * y)
print("Ufunc (np.multiply):\n", np.multiply(x, y))

print("\n=== Division & Square Root ===")
print("Element-wise Division (x / y):\n", x / y)
print("Element-wise Square Root np.sqrt(x):\n", np.sqrt(x))

print("_" * 50)
raw_signals = np.array([-1.7, -0.2, 0.5, 1.4, 2.8])
print("Raw Signals:", raw_signals)
# floor, ceil, and round
print("Floor:", np.floor(raw_signals))
print("Ceil:", np.ceil(raw_signals))
print("Round:", np.round(raw_signals))
print("_" * 50)
print("ceil of 2.4:", np.ceil(2.4))
print("ceil of 2.1:", np.ceil(2.1))
print("ceil of 2.9:", np.ceil(2.9))
print("ceil of -2.9:", np.ceil(-2.9))
print("_" * 50)
print("floor of 2.4:", np.floor(2.4))
print("floor of 2.1:", np.floor(2.1))
print("floor of 2.9:", np.floor(2.9))
print("floor of -2.9:", np.floor(-2.9))
print("_" * 50)

dataset = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Dataset Matrix (3x3):\n", dataset)
# Global Reductions
print("\nGlobal Sum (All elements):", np.sum(dataset))
print("Global Mean:", np.mean(dataset))
print("_" * 50)

# axis=0: Operations collapse rows, computing stats FOR EACH COLUMN
print("\nColumn-wise Sum (axis=0):", np.sum(dataset, axis=0))
print("Column-wise Mean (axis=0):", np.mean(dataset, axis=0))
print("_" * 50)

# axis=1: Operations collapse columns, computing stats FOR EACH ROW
print("\nRow-wise Sum (axis=1):", np.sum(dataset, axis=1))
print("Row-wise Mean (axis=1):", np.mean(dataset, axis=1))
print("_" * 50)
raw_features = np.array([
    [150.0, 2.5, 12.0],
    [170.0, 3.8, 15.0],
    [140.0, 1.9, 10.0],
    [180.0, 4.2, 18.0]
])

col_means = np.mean(raw_features, axis=0)
col_stds = np.std(raw_features, axis=0)
'''
formula for standard deviation: std = sqrt(mean((x - mean(x))^2))
step 1 : compute the mean of each column
step 2 : subtract the mean from each element in the column
step 3 : square the result of step 2
step 4 : compute the mean of the squared differences
'''

# compute standardized features using broadcasting (Z-score normalization)
standardized_features = (raw_features - col_means) / col_stds

print("Raw Features Matrix:\n", raw_features)
print("\nComputed Column Means:", np.round(col_means, 2))
print("Computed Column StdDevs:", np.round(col_stds, 2))
print("\nStandardized Feature Matrix (Mean=0, Std=1):\n", np.round(standardized_features, 3))

print("_" * 50)