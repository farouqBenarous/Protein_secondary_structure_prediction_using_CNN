import keras
import numpy as np
from keras import optimizers, callbacks
from keras.layers import Dropout, Conv1D
from keras.models import Sequential

import dataset

# import keras.backend as K

do_summary = True

LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 20

#loss = 'categorical_hinge' # ok
loss = 'categorical_crossentropy' # best standart
#loss = 'mean_absolute_error' # bad
#loss = 'mean_squared_logarithmic_error' # new best (a little better)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="Whole_CullPDB-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]): # per aminoacid residue
            if np.sum(real[i, j, :]) == 0:  #  real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total


def train_ANN_model():
    print("something")

def run_inference(image, model_version):
    try:
        model = keras.models.load_model('saved_model/' + model_version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()

        model.predict_classes(image)


def test(X, Y, model_version):
    try:
        model = keras.models.load_model('model/alexnet/saved_model/' + model_version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()
        scores = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return scores
