import numpy as np

inch_measures = np.array([1.0, 2.5, 5.0,10.0])
cms_measures = inch_measures * 2.54 # vectorized operation: element-wise multiplication

print("Inches :",inch_measures)
print("CMS :",cms_measures)

feature_a = np.array([1, 2, 3, 4])
feature_b = np.array([10, 20, 30, 40])

print("_" * 60)
print("Feature A:", feature_a)
print("Feature B:", feature_b)
print("Sum of features:", feature_a + feature_b)  # vectorized addition
print("Difference of features:", feature_a - feature_b)  # vectorized subtraction
print("Product of features:", feature_a * feature_b)  # vectorized multiplication
print("Ratio of features:", feature_a / feature_b)  # vectorized division
print("_" * 60)

# Algorithm Rigor
rng = np.random.default_rng(seed=42)
signal_data = rng.uniform(low=0.0, high=1.0, size=(4,4))
print("Signal Data:")
print(signal_data)
# vectorized mask
mask = signal_data > 0.7 # boolean mask for filtering values greater than 0.7
print("Mask:")
print(mask)
print("Filtered Signal Data (values > 0.7):") # data filtering using boolean mask
print(signal_data[mask])  # vectorized filtering using boolean mask
print("_" * 60)

# data type rigor
int_implicit = np.array([1, 2, 3, 4])  # default dtype is int64
int_explicit = np.array([1, 2, 3, 4], dtype=np.int32)  # explicit dtype specification
print("Implicit Integer Array:", int_implicit, "with dtype:", int_implicit.dtype)   
print("Explicit Integer Array:", int_explicit, "with dtype:", int_explicit.dtype)   
print("_" * 60)

# size of items
print("Size of implicit integer array items:", int_implicit.itemsize, "bytes")  # size of each item in bytes
print("Size of explicit integer array items:", int_explicit.itemsize, "bytes")

print("_" * 60)
# numpy charector codes for dtypes
print("Numpy Character Codes for Dtypes:")
print("int64:", np.dtype(np.int64).char)
print("int32:", np.dtype(np.int32).char)
print("float64:", np.dtype(np.float64).char)

# getting types from character codes
print("Character Code 'i' corresponds to dtype:", np.dtype('i'))    
print("Character Code 'f' corresponds to dtype:", np.dtype('f'))
print("Character Code 'd' corresponds to dtype:", np.dtype('d'))
print("Character Code 'f' corresponds to dtype:", np.dtype('f'))
print("Character Code 'f2' corresponds to dtype:", np.dtype('f2'))
print("Character Code 'U10' corresponds to dtype:", np.dtype('U10'))
# U10 means : fixed -length Unicode string of length 10
# i4 means : 32-bit signed integer
print("Character Code 'i4' corresponds to dtype:", np.dtype('i4'))
print("_" * 60)

# custom structured dtype
structured_dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('height', 'f4')])
print("Custom Structured Dtype:", structured_dtype)
print("_" * 60)
employee_data = np.array([('Alice', 30, 5.5), ('Bob', 25, 6.0), ('Charlie', 35, 5.8)], dtype=structured_dtype)
print("Employee Data:")
print(employee_data)

telemetric_schema = np.dtype([('device_id', str,16), ('status_code', np.int32)])
print("Telemetric Schema:", telemetric_schema)