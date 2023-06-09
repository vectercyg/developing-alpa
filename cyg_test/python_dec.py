import jax
import jax.numpy as jnp
import rich.text
from jax.ad_checkpoint import checkpoint_name, print_saved_residuals
from jax.tree_util import tree_flatten, tree_unflatten
from rich.console import Console
from rich.table import Table


def print_fwd_bwd(f, *args, **kwargs) -> None:
    args, in_tree = tree_flatten((args, kwargs))

    def f_(*args):
        args, kwargs = tree_unflatten(in_tree, args)
        return f(*args, **kwargs)

    fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

    y, f_vjp = jax.vjp(f_, *args)
    res, in_tree = tree_flatten(f_vjp)

    def g_(*args):
        *res, y = args
        f_vjp = tree_unflatten(in_tree, res)
        return f_vjp(y)

    bwd = jax.make_jaxpr(g_)(*res, y).jaxpr
    print(fwd)
    print(bwd)

W1 = jnp.ones((5, 4))
W2 = jnp.ones((6, 5))
W3 = jnp.ones((7, 6))
x = jnp.ones(4)



def g(W, x):
  y = jnp.dot(W, x)
  return jnp.sin(y)

# 不使用 jax.checkpoint
def f(W1, W2, W3, x):
  x = g(W1, x)
  x = g(W2, x)
  x = g(W3, x)
  return x

# Inspect the 'residual' values to be saved on the forward pass
# if we were to evaluate `jax.grad(f)(W1, W2, W3, x)`
print("不使用 jax.checkpoint")
# jax.ad_checkpoint.print_saved_residuals(f, W1, W2, W3, x)
# print_fwd_bwd(f, W1, W2, W3, x)

# 使用 jax.checkpoint
print("使用 jax.checkpoint")
def f2(W1, W2, W3, x):
  x = jax.checkpoint(g)(W1, x)
  x = jax.checkpoint(g)(W2, x)
  x = jax.checkpoint(g)(W3, x)
  return x

# jax.ad_checkpoint.print_saved_residuals(f2, W1, W2, W3, x)
print_fwd_bwd(f2, W1, W2, W3, x)

# 选择保存那些值
print("使用 jax.checkpoint，使用 policy 选择保存")
f3 = jax.checkpoint(f, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
# jax.ad_checkpoint.print_saved_residuals(f3, W1, W2, W3, x)
# print_fwd_bwd(f3, W1, W2, W3, x)

# 使用 checkpoint_name 命名中间值
print("使用 jax.checkpoint，使用 checkpoint_name 选择保存")
def f4(W1, W2, W3, x):
  x = checkpoint_name(g(W1, x), name='a')
  x = checkpoint_name(g(W2, x), name='b')
  x = checkpoint_name(g(W3, x), name='c')
  return x

f4 = jax.checkpoint(f4, policy=jax.checkpoint_policies.save_only_these_names('a'))
# jax.ad_checkpoint.print_saved_residuals(f4, W1, W2, W3, x)
