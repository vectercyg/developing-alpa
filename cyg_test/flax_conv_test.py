# 设置jax使用的设备
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import jax
import jax.numpy as jnp
print("device number:{}, device:{}".format(jax.local_device_count(),jax.local_devices()));
import flax.linen as nn
import alpa
class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x

model = CNN()
batch = jnp.ones((32, 64, 64, 10))  # (N, H, W, C) format
variables = model.init(jax.random.PRNGKey(0), batch)
def GetOutput(params,batch):
    output = model.apply(params, batch)
    return output
closed_jaxpr=jax.make_jaxpr(GetOutput)(variables,batch)
print(closed_jaxpr)