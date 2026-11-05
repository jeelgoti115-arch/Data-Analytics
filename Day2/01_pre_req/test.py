import sys
# varchar
max_length = 10000
square_list = [x**2 for x in range(max_length)] #comprehension
#size of
print("List size :",sys.getsizeof(square_list))
print("Sum of list :",sum(square_list))
print("-"*60)
square_gen = (x**2 for x in range(max_length)) #generator
print("Generator size :",sys.getsizeof(square_gen))
print("Sum of Generator :",sum(square_gen))

