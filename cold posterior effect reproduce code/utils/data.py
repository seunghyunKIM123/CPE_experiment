import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds



def get_cifar(name, split='train', num_data=None,data_augmentation=True):

    if name == 'cifar10':
        image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
        image_std = tf.constant([[[0.2470, 0.2435, 0.2616]]])
    elif name == 'cifar100':
        image_mean = tf.constant([[[0.5071, 0.4865, 0.4409]]])
        image_std = tf.constant([[[0.2673, 0.2564, 0.2762]]])

    ds,ds_info = tfds.load(name,
            split=(split if num_data is None else f'{split}[:{num_data}]'),
            as_supervised=True,
            with_info = True)
    
    image_shape = ds_info.features['image'].shape
    
    def preprocess(image, label):
      if data_augmentation:
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
        image = tf.image.random_crop(image, image_shape)

      image = tf.image.convert_image_dtype(image, tf.float32)
      return image, label

    ds = ds.map(preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    ds = ds.batch(num_data or 60000)
    ds = tfds.as_numpy(ds)

    return jax.tree_map(jnp.asarray, next(iter(ds)))

def get_metadata(config):

    metadata = {}

    if config.dataset in ['mnist', 'fashion_mnist']: # 이거 imagenet으로 바꾸기
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
