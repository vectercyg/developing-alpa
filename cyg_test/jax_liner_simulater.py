import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

xs=np.random.normal(size=(100,))
noise=np.random.normal(size=(100,))
ys=xs*3+noise

def model(theta,x):
    w,b=theta
    return w*x+b

def loss_fn(theta,x,y):
    predict=model(theta,x)
    return jnp.mean((predict-y)**2)

@jax.jit
def train(theta,x,y,lr=0.01):
    return theta-lr*jax.grad(loss_fn)(theta,x,y)

theta=np.array([1.,1.])
# for _ in range(1000):
#     theta=train(theta,xs,ys)


# print(theta)
# plt.scatter(xs,ys,c="red")
# plt.plot(xs,model(theta,xs))
# plt.savefig("/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/cyg_test/liner_simulator.png")

print(jax.make_jaxpr(train)(theta,xs,ys))
# print(jax.make_jaxpr(jax.jit(train))(x,y))