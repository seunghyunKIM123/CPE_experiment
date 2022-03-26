import jax
import jax.numpy as jnp

def schedule(epoch): # for swa
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


def make_cyclical_cosine_lr_schedule(init_lr, num_batches, cycle_epochs, const_epochs=0):

    cycle_steps = cycle_epochs * num_batches

    def schedule(step):
        t = jnp.maximum(step - const_epochs * num_batches - 1, 0)
        t = (t % cycle_steps) / cycle_steps
        return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

    return schedule

def normal_like_tree(a, key, std=1.0):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_multimap(lambda p, k: std*jax.random.normal(k, shape=p.shape),
            a, jax.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]

def uniform_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_multimap(lambda p, k: jax.random.uniform(k, shape=p.shape,minval=-1/jnp.sqrt(total_params(p)),maxval=1/jnp.sqrt(total_params(p))),
            a, jax.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]


def batch_divider(perm_key,batch_size,num_data):
    batch_indices = jax.random.permutation(perm_key, jnp.arange(num_data))
    quotient = num_data//batch_size
    batch_indices = batch_indices[:quotient*batch_size]
    if num_data % batch_size != 0:
        batch_indices = jax.tree_map(lambda x: x.reshape((quotient, batch_size)), batch_indices)
    else:
        batch_indices = jax.tree_map(lambda x: x.reshape((quotient,batch_size)),batch_indices)

    return batch_indices

def tree_add(a, b):
  return jax.tree_multimap(lambda e1, e2: e1+e2, a, b)



