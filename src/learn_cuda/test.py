
import timeit
import cv2
set_stmt = """
import numpy as np
a = [
    [64.000000, 84.000000, 1.000000, 0.000000, 0.000000, 0.000000], 
    [64.000000, 20.000000, 1.000000, 0.000000, 0.000000, 0.000000], 
    [0.000000, 20.000000, 1.000000, 0.000000, 0.000000, 0.000000], 
    [0.000000, 0.000000, 0.000000, 64.000000, 84.000000, 1.000000], 
    [0.000000, 0.000000, 0.000000, 64.000000, 20.000000, 1.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 20.000000, 1.000000]
    ]
b = [240.000000, 
240.000000, 
-16.000000, 
320.000000, 
64.000000, 
64.000000]
"""
stmt = """
x = np.linalg.solve(a, b)
"""
print(timeit.timeit(stmt, setup=set_stmt, number=1000))
