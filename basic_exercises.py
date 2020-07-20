import numpy as np
import os
import matplotlib.pyplot as plt

# Write a NumPy program to get the numpy version and show numpy build configuration.
print(np.__version__)
print(np.show_config())


# Write a NumPy program to  get help on the add function
# Note: np.info() Get help information for a function, class, or module.
print(np.info(np.add))


# Write a NumPy program to test whether none of the elements of a given array is zero
# Note: np.all() evaluates all the values are true in the given array
a = np.array([1, 2, 4])
print(np.all(a))


# Write a NumPy program to test whether any of the elements of a given array is non-zero.
# Note: np.any() Test whether any array element along a given axis evaluates to True.
a = np.array([1, 2, 3, 5])
print(np.any(a))


# Write a NumPy program to test a given array element-wise for finiteness (not infinity or not a Number)
# Note: np.isfinite() Test element-wise for finiteness (not infinity or not Not a Number).
a = np.array([1, np.nan, 2, np.inf])
print(np.isfinite(a))

# Write a NumPy program to test element-wise for positive or negative infinity.
# Note: np.isnf() Test element-wise for positive or negative infinity.
a = np.array([1, np.nan, 2, np.inf])
print(np.isinf(a))


# Write a NumPy program to test element-wise for NaN of a given array.
# Note: np.isnan() Test element-wise for NaN and return result as a boolean array.
a = np.array([1, np.nan, 345])
print(np.isnan(a))


# Write a NumPy program to test element-wise for complex number, real number of a given array. Also test whether a given number is a scalar type or not.
a = np.array([1, 2+0.1j, 3, 4])
print(np.iscomplex(a))
print(np.isreal(a))
print(np.isscalar(a))


# Write a NumPy program to test whether two arrays are element-wise equal within a tolerance.
a = np.array([1, 2, 4, 5])
b = np.array([1, 2, 4, 5])
print(np.allclose(a, b))
print(np.allclose([1, np.nan], [1, np.nan]))
print(np.allclose([1, np.nan], [1, np.nan], equal_nan=True))


# Write a NumPy program to create an element-wise comparison (greater, greater_equal, less and less_equal) of two given arrays.
x = np.array([1, 3])
y = np.array([2, 4])
print(np.greater(x, y))
print(np.greater_equal(x, y))
print(np.less(x, y))
print(np.less_equal(x, y))


# Write a NumPy program to create an array with the values 1, 7, 13, 105 and determine the size of the memory occupied by the array.
x = np.array([1, 7, 13, 105])
print('%d bytes ' %(x.size * x.itemsize))


# Write a NumPy program to create an array of 10 zeros,10 ones, 10 fives.
print(np.ones(10))
print(np.zeros(10))
print(np.ones(10)*5)


# Write a NumPy program to create an array of the integers from 30 to 70.
a = np.arange(30, 71)
print(a)


# Write a NumPy program to create an array of all the even integers from 30 to 70
a = np.arange(30, 71, 2)
print(a)


# Write a NumPy program to create a 3x3 identity matrix.
print(np.identity(3))


#  Write a NumPy program to generate a random number between 0 and 1.
print(np.random.normal(0, 1, 1))


# Write a NumPy program to generate an array of 15 random numbers from a standard normal distribution
a = np.random.normal(0, 1, 15)
print(a)


# # Write a NumPy program to create a vector with values ranging from 15 to 55 and print all values except the first and last.
a = np.arange(15, 56)
print(a[1:-1])


# Write a NumPy program to create a 3X4 array using and iterate over it.
a = np.arange(10, 22).reshape((3, 4))
for x in np.nditer(a):
    print(x, end=' ')


#  Write a NumPy program to create a vector of length 10 with values evenly distributed between 5 and 50.
print(np.linspace(5, 49, 10))


# Write a NumPy program to create a vector with values from 0 to 20 and change the sign of the numbers in the range from 9 to 15.
x = np.arange(21)
x[(x >= 9) & (x <= 15)] *= -1
print(x)


#  Write a NumPy program to create a vector of length 5 filled with arbitrary integers from 0 to 10.
print(np.random.randint(0, 11, 5))


#  Write a NumPy program to multiply the values of two given vectors.
a = np.array([12, 23, 45])
b = np.array([1, 3, 5])
print(a * b)


# Write a NumPy program to create a 3x4 matrix filled with values from 10 to 21.
print(np.arange(10, 22).reshape((3, 4)))


# Write a NumPy program to find the number of rows and columns of a given matrix.
a = np.arange(10, 22).reshape((3, 4))
print(a.shape)


# Write a NumPy program to create a 3x3 identity matrix, i.e. diagonal elements are 1, the rest are 0.
print(np.eye(3))


# Write a NumPy program to create a 10x10 matrix, in which the elements on the borders will be equal to 1, and inside 0
a = np.ones((10, 10))
a[1:-1, 1:-1] = 0
print(a)


# Write a NumPy program to create a 5x5 zero matrix with elements on the main diagonal equal to 1, 2, 3, 4, 5.
a = np.diag([1, 2, 3, 4, 5])
print(a)


# Write a NumPy program to create a 4x4 matrix in which 0 and 1 are staggered, with zeros on the main diagonal.
a = np.zeros((4,4))
a[::2, 1::2] = 1
a[1::2, ::2] = 1
print(a)


# Write a NumPy program to create a 3x3x3 array filled with arbitrary values.
print(np.random.random((3, 3, 3)))


# Write a NumPy program to compute sum of all elements, sum of each column and sum of each row of a given array.
a = np.array(((1, 2, 3, 4), (1, 2, 3, 4)))
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))


# Write a NumPy program to compute the inner product of two given vectors.
a = np.array([1, 2, 3, 5])
b = np.array([1, 2, 3, 5])
print(np.dot(a, b))


# Write a NumPy program to add a vector to each row of a given matrix
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 1, 0])
result = np.empty_like(m)
for i in range(4):
    result[i, :] = m[i, :] + v
print(result)


# Write a NumPy program to save a given array to a binary file.
a = np.arange(20)
np.save('temp_array.npy', a)
if os.path.exists('temp_array.npy'):
    x2 = np.load('temp_array.npy')
    print(np.array_equal(a, x2))


# Write a NumPy program to save two given arrays into a single file in compressed format (.npz format) and load it.
a = np.arange(10)
b = np.arange(11, 20)
np.savez('temp.npz', x=a, y=b)
with np.load('temp.npz') as data:
    a2 = data['x']
    b2 = data['y']
    print(a2)
    print(b2)


# Write a NumPy program to save a given array to a text file and load it.
a = np.arange(12).reshape(4, 3)
header = 'A B C'
np.savetxt('temp_ar.txt', a, fmt='%d', header=header)
print(np.loadtxt('temp_ar.txt'))


# Write a NumPy program to convert a given array into bytes, and load it as array.
a = np.arange(10)
a_bytes = a.tobytes()
a2 = np.frombuffer(a_bytes, dtype=a.dtype)
print(a2)


# Write a NumPy program to convert a given array into a list and then convert it into a list again.
a = [1, 2, 3, 4]
a_np = np.array(a)
print(a_np)
print(a_np.tolist())
print(a == a_np)


# Write a NumPy program to compute the x and y coordinates for points on a sine curve and plot the points using matplotlib.
x = np.arange(0, 3 * np.pi, 0.2)
y = np.sin(x)
plt.plot(x, y)
plt.show()


# Write a NumPy program to convert numpy dtypes to native python types.
x = np.float32(0)
print(type(x))
y = x.item()
print(type(y))
