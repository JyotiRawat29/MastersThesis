#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 00:07:46 2021

@author: jyoti
"""

import math 


softmax_output =[0.7, 0.1, 0.2] # just consider this is the output you have got from the softmax activation function of the output layer
target_output =[1,0,0]
target_class = 0
loss = -(math.log(softmax_output[0]) * target_output[0] +
            math.log(softmax_output[1]) * target_output[1] +
            math.log(softmax_output[2]) * target_output[2])
print(loss)
loss = -math.log(softmax_output[0])
print(loss)
#loss is higher if we wrong class and loss is lesser if we take the right class