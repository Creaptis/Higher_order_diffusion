import jax
import jax.numpy as jnp

f = lambda x, y: (x+1, x * y)
xs = jnp.arange(10)
init = 0
out = jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1)

print(len(out))
print([out[i].shape for i in range(len(out))])
print(out)