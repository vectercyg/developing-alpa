import numpy as np  
import ray
# Define two remote functions. Invocations of these functions create tasks  
# that are executed remotely.  
# @ray.remote  
def multiply(x, y):  
    return np.dot(x, y)  
# @ray.remote  
def zeros(size):  
    return np.zeros(size)  
ray.init(address="auto")
# # Start two tasks in parallel. These immediately return futures and the  
# # tasks are executed in the background.  
# x_id = zeros.remote((100, 100))  
# y_id = zeros.remote((100, 100))  
# # Start a third task. This will not be scheduled until the first two  
# # tasks have completed.  
# z_id = multiply.remote(x_id, y_id)  
# # Get the result. This will block until the third task completes.  
# z = ray.get(z_id)  
# print(z)     
cls=ray.remote(num_cpus=1)(zeros)
cls.options
x_id=cls.remote()
cls=ray.remote(num_cpus=1)(multiply)
pass