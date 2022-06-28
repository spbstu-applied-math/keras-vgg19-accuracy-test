import tensorflow as tf
import tensorflow_datasets as tfds

from keras.applications.vgg19 import preprocess_input as vgg19_preprocess


def center_crop_resize(image, shape):
    target_width = shape[0]
    target_height = shape[1]
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]
    im = image

    initial_ratio = initial_width / initial_height
    target_ratio = target_width / target_height

    w = target_width
    h = target_height
    if target_ratio > initial_ratio:
        ratio = tf.cast(target_width / initial_width, tf.float32)
        h = tf.cast(tf.cast(initial_height, tf.float32) * ratio, tf.int32)
        if h < target_height:
            h = target_height
    else:
        ratio = tf.cast(target_height / initial_height, tf.float32)
        w = tf.cast(tf.cast(initial_width, tf.float32) * ratio, tf.int32)
        if w < target_width:
            w = target_width
    
    im = tf.image.resize(im, (w, h), method="bicubic")
    
    width = tf.shape(im)[0]
    height = tf.shape(im)[1]
    startx = (width//2 - target_width//2)
    starty = (height//2 - target_height//2)
    im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
    return im


def vgg19_resize(image):
    return center_crop_resize(image, shape=(224,224))


def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = vgg19_resize(image)
    image = vgg19_preprocess(image)
    return image


def load_test_dataset(data_dir, batch_size):
    dataset = tfds.load(name="imagenet_v2",
                            split='test[:25%]',
                            as_supervised=True,
                            download=False, 
                            data_dir=data_dir)
    dataset = dataset.map(lambda i, l: (preprocess_image(i), l)).batch(batch_size)
    return dataset
