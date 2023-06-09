# from jax.experimental import mesh_utils
# from jax.experimental.maps import Mesh
# from jax.interpreters.pxla import PartitionSpec,ShardedDeviceArray,ShardingSpec
# from jax.experimental.pjit import MeshPspecSharding,PmapSharding
# from jax.sharding import MeshPspecSharding,SingleDeviceSharding
# # from jax.interpreters.xla import device_put
# import jax
# jax.config.update('jax_array', True)
# import time
# devices=mesh_utils.create_device_mesh((2,1))
# mesh=Mesh(devices,('x','y'))
# sharding=MeshPspecSharding(mesh,PartitionSpec('x','y'))
# print(type(sharding))
# a=jax.numpy.ones((256,256))
# a_=jax.device_put(a,sharding)
# # jax.debug.visualize_array_sharding(a_)
# print(type(a))
# # sharding=MeshPspecSharding(mesh,PartitionSpec('y','x'))
# # b_=jax.device_put(a,sharding)
# # # jax.debug.visualize_array_sharding(b_)

# # # c=jax.numpy.dot(a_,b_)
# c=jax.jit(lambda x: x * 2)(a_)
# # c=jax.numpy.sin(a_)
# # print(c)
import jax
from jax import nn as nn
from jax import numpy as np

W=np.ones((2,2),np.float32)
B=np.ones((2,1),np.float32)
x=1.
jax.disable_jit()
def FC(x):
    return x*2.
# # print(nn.leaky_relu(2,3))
jaxpr=jax.jvp(FC,(1.,),(2.,))
print(jaxpr)
# # print(jaxpr(2,3))

def f(x):
    return x+1

a=np.ones((3))
print(a)
b=jax.jit(f)
b=b(a)
print(b)