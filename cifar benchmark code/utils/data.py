import jax
import jax.numpy as jnp
jnp.ones(1)
import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices([], "GPU")
  except RuntimeError as e:
    # tensor flow에서 gpu 메모리를 쓰지 못하게 하고 시작해야된다
    print(e)


def get_cifar(name, split='train', num_data=None):

    if name == 'cifar10':
        image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
        image_std = tf.constant([[[0.2470, 0.2435, 0.2616]]])
    elif name == 'cifar100':
        image_mean = tf.constant([[[0.5071, 0.4865, 0.4409]]])
        image_std = tf.constant([[[0.2673, 0.2564, 0.2762]]])

    ds = tfds.load(name,
            split=(split if num_data is None else f'{split}[:{num_data}]'),
            as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - image_mean) / image_std
        return image, label

    ds = ds.map(preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    ds = ds.batch(num_data or 60000)
    ds = tfds.as_numpy(ds)

    return jax.tree_map(jnp.asarray, next(iter(ds)))

def get_metadata(config):

    metadata = {}

    if config.dataset in ['mnist', 'fashion_mnist']:
        metadata['num_train'] = 60000
        metadata['num_test'] =  10000
        metadata['num_classes'] = 10
        metadata['shape'] = (28, 28, 1)
    elif config.dataset in ['cifar10', 'cifar100']:
        metadata['num_train'] =  50000
        metadata['num_test'] =  10000
        metadata['num_classes'] = 10 if config.dataset == 'cifar10' else 100
        metadata['shape'] = (32, 32, 3)
    else:
        raise ValueError(f'Invalid data {config.dataset}')


    metadata['num_train_batches'] = metadata['num_train'] // config.batch_size
    metadata['num_test_batches'] = metadata['num_test'] // config.batch_size

    return metadata
