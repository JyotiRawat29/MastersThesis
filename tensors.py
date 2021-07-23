#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:43:17 2021

@author: jyoti
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype = torch.float32,
                         requires_grad = True)
print(my_tensor)

print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.dtype)


#other common initialisation methods
x = torch.empty(size = (3,3)) #creates a tensor with set of random values
print(x)

x = torch.zeros((3,3))
print(x)

x = torch.rand((3,3)) # uniform distribution with values between 0 and 1
print(x)

x = torch.ones((3,3))
print(x)

x = torch.eye(3,3) #identity matrix
print(x)

x = torch.arange(start = 0, end = 5, step =1)
print(x)

x = torch.linspace(start=0.1, end = 1, steps = 10) # 10 vlaues between 0.1 and 1

x = torch.empty(size = (1,5)).normal_(mean = 0, std =1)
x = torch.empty(size = (1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))
print(x)

#intialise and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

#array to tensor conversion and vice versa
import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()

#Tensor Math and comparision operation
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x,y, out = z1) #addition
print(z1)

z2 = torch.add(x,y)
print(z2)
z = x+y
print(z)

#substraction
z = x-y

#division
z = torch.true_divide(x, y)
print(z)

#inplace operation
t = torch.zeros(3)
t.add_(x)
t+= x # t= t+x 

#exponentiation
z = x.pow(2)
print(z)
z = x**2

#comparision
z = x > 0
print(z)
z = x < 0

#Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

#Matrix Exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)
print(matrix_exp.matrix_power(3)) #multiplying matrix 3 times by itself

#element wise multiplication
z = x*y
print(z)

#dot product
z = torch.dot(x,y)
print(z)

#Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m)) # when we have 3 dimension multiplcation we do batch multiplication
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1, tensor2) #m should be same in both tensors in order to do multiplication i.e. the 3rd element of 1st tensor  and the 2nd element of the 2nd tensor should be same
#(batch,n,p)

#example of braoadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 -x2
z = x1 ** x2

#other useful operations
sum_x = torch.sum(x, dim = 0)
values, indices = torch.max(x, dim = 0) # x.max(dim,0)
values, indices = torch.min(x, dim = 0)
abs_x = torch.abs(x)
z = torch.abs(x)
z = torch.argmax(x, dim = 0)
z = torch.argmin(x, dim = 0)
mean_x = torch.mean(x.float(), dim = 0)
z = torch.eq(x,y)
print(z)
torch.sort(y, dim =0, descending = False)
#or
sorted_y, indices = torch.sort(y, dim =0, descending = False)
z = torch.clamp(x, min = 0) #if thereis any value less than it will clamp all those values

x = torch.tensor([1,0,1,1,1], dtype = torch.bool)
z = torch.any(x)
z = torch.all(x)

#========================================#
#           TENSOR INDEXING           #
#========================================#

batch_size = 10
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape)
print(x[:,0].shape)

print(x[2,0:10]) #0;10 --> [0,1,2,........,9]
x[0,0] = 100

#Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols].shape)

#more advanced indexing

x = torch.arange(10)
print(x[(x<2) & (x>8)])
print(x[x.remainder(2)== 0])

#useful operations
print(torch.where(x>5,x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension()) #5x5x5
print(x.numel())


#=========================================================#
#           RESHAPING TENSOR                              #
#=========================================================#

x = torch.arange(9)
x_3x3 = x.view(3,3)
print(x_3x3)
x_3x3 = x.reshape(3,3)
print(x_3x3) #view and reshape are similiar in many ways but view act on contguious tensors means memory block should be stored in contagious manner
# hence reshape is better but performance loss
y = x_3x3.t()
print(y)
#print(y.view(9))# will be erreneous hence use
# view is used to flatten the tensor generally it is used as view(-1) whene we want to flat a n-dimension tensor e.g. of sie (2,5,2) ; the size after flattening of the tensor with view(-1) would be (20)
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim =0).shape)
print(torch.cat((x1,x2), dim =1).shape)
z = x1.view(-1)
print(z.shape)

#EXAMPLE
batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch, -1)
print(z.shape)

#if we want to keep the batch but switch 2,5
z = x.permute(0,2,1) #we want dimension 2 at place 1 and dimension at place 1, here(0,2,1) 2 and 1 are the places of the dimension and not the numbers
print(z.shape)

x = torch.arange(10) 
print(x.unsqueeze(0).shape) # to make shape 1,10
print(x.unsqueeze(1).shape) # to make shaoe 10,1

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # to make shape 1x1x10 i.e. 1x10

z = x.squeeze(1) #shape 1x10
print(z.shape)