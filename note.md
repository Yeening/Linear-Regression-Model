# Note of implementation

## Numpy

### Initialize

```python
# Initial with range
square = np.array(range(4)).reshape(2,2)
# Initial with array
x = np.array([1,2,3],[4,5,6])
a = [1,2,3]
y = np.array(a)
# Initial with zeros
thetas = np.zeros(x[1].shape,dtype=float)
```

### Square and power

```python
# Square
arr1 = [1, -3, 15, -466] 
np.square(arr1) # [1,9,225,217156]
# Power
x1 = range(6) # [0, 1, 2, 3, 4, 5]
np.power(x1, 3) # array([  0,   1,   8,  27,  64, 125])
x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
np.power(x1, x2) # array([  0.,   1.,   8.,  27.,  16.,   5.])
```

### Sum

```python
# 1D array  
arr = [20, 2, .2, 10, 4]  
print("\nSum of arr : ", np.sum(arr)) # Sum of arr :  36.2
print("Sum of arr(uint8) : ", np.sum(arr, dtype = np.uint8)) # Sum of arr(uint8) :  36
print("Sum of arr(float32) : ", np.sum(arr, dtype = np.float32)) # Sum of arr(float32) :  36.2
# 2D array
arr = [[14, 17, 12, 33, 44],    
       [15, 6, 27, 8, 19],   
       [23, 2, 54, 1, 4,]]
print("\nSum of arr : ", np.sum(arr))  
print("Sum of arr(axis = 0) : ", np.sum(arr, axis = 0))  
print("Sum of arr(axis = 1) : ", np.sum(arr, axis = 1)) 
# Sum of arr :  279
# Sum of arr(axis = 0) :  [52 25 93 42 67]
# Sum of arr(axis = 1) :  [120  75  84]
```



## Numpy Linear Algebra

### Inner Product and Transpose

```python
x, y = load_data_set(filename)
print(x.shape, x.T.shape) #(200, 2) (2, 200)
np.dot(x.T,x).shape #(2, 2)
# Another method
x.T.dot(x),shape #(2, 2)
square = np.array(range(4)).reshape(2,2)
np.dot(square, square, square) #also accept more than two parameters
```

* Caution with the dtype of the arrays that you want to make dot. you can transfer it to `float` in linear regression:

  ```python
  np.array(x,dtype=float), np.array(y,dtype=float)
  ```

* np.dot vs. a.dot(b)

  ```python
  np.dot(inv_xTx,x.T).dot(y) #It works
  np.dot(inv_xTx,x.T,y) 
  #ValueError: output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)
  ```

### Inverse Matrix

`np.linalg.inv(*target_matrix*)`

```python
square = np.array([
    [2,3],
    [1,4]
])
print(np.linalg.inv(square))
# [[ 0.8 -0.6]
#  [-0.2  0.4]]
print(np.dot(np.linalg.inv(square),square))
# [[1. 0.]
#  [0. 1.]]
```

## Python random

```python
import random 
print (random.choice([1, 4, 8, 10, 3])) # A random number from list
random.randrange(20, 50, 3) # A random number from range(20,50), with the gap of 3
```

