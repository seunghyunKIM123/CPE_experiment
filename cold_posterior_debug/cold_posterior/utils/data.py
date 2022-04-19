import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.utils.data import DataLoader
'''
#https://github.com/google-research/google-research/blob/master/cold_posterior_flax/cifar10/input_pipeline.py 참고
def get_train_cifar(dataset :str, data_augmentation :int):

    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
    else:
        train_transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
    CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
    train_dataset = CIFAR(root="~/datasets", train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True, num_workers=4)
    return train_loader


def get_test_cifar(dataset: str, data_augmentation: int):
    if dataset == "cifar10" or dataset == "cifar100":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        valid_dataset = CIFAR(root="~/datasets", train=False, download=True, transform=valid_transform)
        valid_loader = DataLoader(valid_dataset, batch_size=10000, shuffle=False, num_workers=4)

        return jax.tree_map(jnp.asarray, next(iter(valid_loader)))
'''
def get_cifar(name, data_augmentation : int, split : str , num_data=None):

  #  if name == 'cifar10':
  #      image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
  #      image_std = tf.constant([[[0.2470, 0.2435, 0.2616]]])
  #  elif name == 'cifar100':
  #      image_mean = tf.constant([[[0.5071, 0.4865, 0.4409]]])
  #      image_std = tf.constant([[[0.2673, 0.2564, 0.2762]]])

    ds,ds_info = tfds.load(name,
            split=(split if num_data is None else f'{split}[:{num_data}]'),
            as_supervised=True,
            with_info = True)
    
    image_shape = ds_info.features['image'].shape
    
    def preprocess(image, label):
      image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
      image_std =  tf.constant([[[0.2470, 0.2435, 0.2616]]])
   #   else:
   #       image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
   #       image_std = tf.constant([[[0.2470, 0.2435, 0.2616]]])

      if data_augmentation==1:
          image = tf.pad(image, [[2, 2], [2, 2], [0, 0]],'CONSTANT') # crop_padding = 4 , constant? REFLECT? 아 ㅅㅂ 왜  성능이 떨어지노
          image = tf.image.random_crop(image, image_shape) # crop
          image = tf.image.random_flip_left_right(image)
      else:
          pass

      image = tf.cast(image, tf.float32) / 255.0 # 이미 scale 이 0,1사이 scale로 되어있는거같은데?
      image = (image - image_mean) / image_std

      return image, label

    ds = ds.map(preprocess)
    ds = ds.batch(60000)
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

