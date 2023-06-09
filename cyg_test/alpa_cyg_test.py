"""
.. _alpa-quickstart:

Alpa Quickstart
===============

Alpa is built on top of a tensor computation framework `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ .
Alpa can automatically parallelize jax functions and runs them on a distributed cluster.
Alpa analyses the computational graph and generates a distributed execution plan
tailored for the computational graph and target cluster.
The generated execution plan can combine state-of-the-art distributed training techniques
including data parallelism, operator parallelism, and pipeline parallelism.

Alpa provides a simple API ``alpa.parallelize`` and automatically generates the best execution
plan by solving optimization problems. Therefore, you can efficiently scale your jax computation
on a distributed cluster, without any expertise in distributed computing.

In this tutorial, we show the usage of Alpa with an MLP example.
"""

################################################################################
# Import Libraries
# ----------------
# We first import the required libraries.
# Flax and optax are libraries on top of jax for training neural networks.
# Although we use these libraries in this example, Alpa works on jax's and XLA's internal
# intermediate representations and does not depend on any specific high-level libraries.

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random

import alpa
from alpa.testing import assert_allclose

# import cyg_test.display.jaxpr_display as jaxpr_display
################################################################################
# Train an MLP on a Single Device
# -------------------------------
# To begin with, we implement the model and training loop on a single device. We will
# parallelize it later. We train an MLP to learn a function y = Wx + b.


class MLPModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            if i % 2 == 0:
                x = nn.Dense(features=self.hidden_dim * 4)(x)
            else:
                x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        # I=jnp.ones_like(x,jnp.float32)
        # x = nn.Dense(features=self.hidden_dim)(x)+I
        # out1 = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(out1)+I
        # out2 = nn.relu(x)
        # x=out1+out2
        
        # x = nn.Dense(features=self.hidden_dim * 4)(x) 
        # x = nn.Dense(features=self.hidden_dim)(x)
        return x


# 分片并行
method=alpa.ShardParallel(num_micro_batches=16)
@alpa.parallelize(method=method)
def alpa_shard_train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

# 流水线分片并行
method = alpa.PipeshardParallel(num_micro_batches=16,
                            layer_option=alpa.AutoLayerOption(layer_num=2),
                            stage_option="auto")


@alpa.parallelize(method=method)
def alpa_pipe_train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        # loss = jnp.mean((out - batch["y"])**2)
        loss=out
        # temp=loss+5
        return loss

    grads = alpa.grad(loss_func)(state.params)
    # value,grads=alpa.value_and_grad(loss_func)(state.params)
    # value = value+5
    # print(value)
    # closed_jaxpr=jax.make_jaxpr(alpa.grad(loss_func))(state.params)
    # print(closed_jaxpr)

    new_state = state.apply_gradients(grads=grads)
    return new_state
    
    
def train(state,batch,parallelize_fun=alpa_shard_train_step):
    for i in range(1):
        state=parallelize_fun(state,batch)
    return state

def getStateBatchModel(dim=2048,batch_size=2048,num_layers=10,is_batch=False):

    # Generate ground truth W and b
    rngkey = jax.random.PRNGKey(0)
    k1, k2 = random.split(rngkey)
    # W = random.normal(k1, (dim, dim))
    # b = random.normal(k2, (dim,))

    # Generate the training data
    ksample, knoise = random.split(k1)
    if is_batch:
        x = random.normal(ksample, (dim,))
        x = jax.numpy.stack([x for i in range(batch_size)])
        y = (x * 2 + 1)
    else:
        x = random.normal(ksample, (1, dim))
        y = (x * 2 + 1)

    # Initialize a train state, which includes the model paramter and optimizer state.
    model = MLPModel(hidden_dim=dim, num_layers=num_layers)
    params = model.init(rngkey, x)
    tx = optax.adam(learning_rate=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    batch = {"x": x, "y": y}
    return state,batch,model
    state=alpa_pipe_train_step(state,batch)

from alpa.util import benchmark_func


def alpa_execution(parallelize_fun=alpa_shard_train_step):
    global state,batch
    state = parallelize_fun(state, batch)
    
def getAlpaParallelizeData(state,batch,executable=None,parallelize_fun=alpa_shard_train_step,pipe_layer=2):
    def alpa_execution(parallelize_fun=parallelize_fun):
        nonlocal state,batch
        state = parallelize_fun(state, batch)
        
    def shard_parallelize_sync_func():
        jax.local_devices()[0].synchronize_all_activity()
    
    def pipe_parallelize_sync_func():
        executable.sync()
    
    if executable is None:
        alpa_costs = benchmark_func(alpa_execution, shard_parallelize_sync_func, warmup=5, number=10, repeat=5) * 1e3
        print(f"Alpa execution time.   Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms")
        GB = 1024 ** 3
        executable = parallelize_fun.get_executable(state, batch)
        print(f"Alpa execution per GPU memory usage:   {executable.get_total_allocation_size() / GB:.2f} GB")
    else:
        alpa_costs = benchmark_func(alpa_execution, pipe_parallelize_sync_func, warmup=5, number=10, repeat=5) * 1e3   
        print(f"Alpa execution time.   Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms")
        stage_allocation_size=executable.get_stage_allocation_size()
        stage_list=list()
        GB = 1024 ** 3
        for _ in range(pipe_layer):
            stage_list.append(0)
        for i in range(len(stage_allocation_size)):
            stage_list[i%pipe_layer]+=stage_allocation_size[i]
        print(f"Alpa execution per GPU memory usage:   {['%.8f'%(stage_size / GB) for stage_size in stage_list]} GB")
        
        

    
    
    
def draw(x,y,model,state):
    import matplotlib.pyplot as plt
    plt.scatter(x[0],y[0],c="red")
    # print(state.apply_fn(state.params,batch["x"]))
    y=model.apply(state.params,batch["x"])
    plt.scatter(x[0],y[0])
    plt.savefig("/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/cyg_test/liner_simulator.png")

if __name__ == '__main__':
    import os

    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = 0.80
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]= "platform"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # shard 并行
    state,batch,model=getStateBatchModel(num_layers=4,is_batch=True)
    state=train(state,batch,parallelize_fun=alpa_shard_train_step)
    # draw(batch["x"],batch["y"],model,state)
    
    # state,batch,model=getStateBatchModel(num_layers=10,is_batch=True)
    # # Benchmark parallel execution with alpa
    # # We distribute arguments in advance for the benchmarking purpose.
    # state, batch = alpa_shard_train_step.preshard_dynamic_args(state, batch)
    # getAlpaParallelizeData(state, batch)
    
    # # pipeshard并行
    # alpa.init(cluster="ray")
    # state,batch,model=getStateBatchModel(num_layers=2,is_batch=True)
    # state=train(state,batch,parallelize_fun=alpa_pipe_train_step)
    # draw(batch["x"],batch["y"],model,state)
    
    # alpa.init(cluster="ray")
    # state,batch,model=getStateBatchModel(num_layers=12,is_batch=True)
    # # # state, batch = alpa_pipe_train_step.preshard_dynamic_args(state, batch)
    # state=alpa_pipe_train_step(state,batch)
    # executable=alpa_pipe_train_step.get_last_executable()
    # getAlpaParallelizeData(state, batch,executable,alpa_pipe_train_step)
    
    # # 测试
    # alpa.init(cluster="ray")
    # state,batch,model=getStateBatchModel(dim=10,is_batch=True)
    # state=alpa_pipe_train_step(state,batch)
    # executable=alpa_pipe_train_step.get_last_executable()
    # executable=alpa_shard_train_step.get_last_executable()
    # costs = []
    # # Warmup
    # for _ in range(5):
    #     # model.apply(state.params,batch["x"])
    #     state=alpa_shard_train_step(state,batch)
    # import time
    # # Choose a "number" according to "min_repeat_second"
    # # Benchmark
    # for _ in range(5):
    #     if True:
    #         # executable.sync()
    #         jax.local_devices()[0].synchronize_all_activity()
    #     tic = time.time()
    #     for _ in range(10):
    #         alpa_shard_train_step(state,batch)
    #         # model.apply(state.params,batch["x"])
    #     if True:
    #         # executable.sync()
    #         jax.local_devices()[0].synchronize_all_activity()
    #     costs.append(time.time() - tic)

    # print(np.array(costs) / 5)
    
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # from pycallgraph.config import Config
    # from pycallgraph.globbing_filter import GlobbingFilter
    # tracker_log="cyg_test/trace.dot"
    # output_file="cyg_test/pycallgraph.png"
    # dot_file_path="cyg_test/pycallgraph.dot"
    # trace_filter = GlobbingFilter(
    #             exclude=['pycallgraph.*',
    #             'pandas.*',
    #             'numpy.*',
    #             'jax.*',
    #             'pub.*',
    #             'optax.*'
    #             'featuretools.demo.mock_customer.load_mock_customer',
    #             'featuretools.demo.mock_customer.<listcomp>',
    #             ],
    #             include=['alpa.*',],
    #         )
            
    # with PyCallGraph(output=GraphvizOutput(output_file=output_file,dot_file_path=dot_file_path), config=Config(verbose=True,tracker_log=tracker_log,trace_filter=trace_filter)):
    #     alpa.init(cluster="ray")
    #     state,batch,model=getStateBatchModel(dim=4,is_batch=True)
    #     # # state, batch = alpa_pipe_train_step.preshard_dynamic_args(state, batch)
    #     state=alpa_pipe_train_step(state,batch)




