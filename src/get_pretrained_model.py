from keras.applications.vgg19 import VGG19
from keras.models import Model


if __name__ == '__main__':

    model: Model = VGG19(weights='imagenet')

    model.compile(loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

    model.save('data/model.h5', save_format='h5')
