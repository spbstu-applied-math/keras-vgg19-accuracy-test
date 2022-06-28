import json

from keras.models import Model, load_model
from load_test_dataset import load_test_dataset


if __name__ == '__main__':

    BATCH_SIZE = 16

    model: Model = load_model('data/model.h5')

    test_dataset = load_test_dataset('data', BATCH_SIZE)

    result = model.evaluate(test_dataset)

    json.dump(
        obj={
            'accuracy': result[1],
            'top_5_accuracy': result[2]
        },
        fp=open('data/eval.json', 'w')
    )
