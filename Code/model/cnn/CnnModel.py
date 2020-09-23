import keras
from keras.callbacks import TensorBoard  # for part 3.5 on TensorBoard
from keras.layers import Conv1D
from keras.layers import Dense, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def train_CNN_model_V1(X, Y, dataset_meta_data):
    model = Sequential()

    model.add(Conv1D(128, 11, padding='same', activation='relu',
                     input_shape=(dataset_meta_data['sequence_len'], dataset_meta_data['amino_acid_residues'])))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 11, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(BatchNormalization())

    model.add(BatchNormalization())

    # model.add(Dense(1, activation='softmax'))
    model.add(Conv1D(dataset_meta_data["num_classes"], 11, padding='same', activation='softmax'))  #
    model.add(Dense(1, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    tensor_board = TensorBoard('tensorboard/')

    model.fit(X, Y, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle=True,
              callbacks=[tensor_board])

    model.save('saved_model/cnnV1.h5', include_optimizer='true')


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
        model = keras.models.load_model('saved_model/' + model_version)
    except (ImportError, IOError) as error:
        print("Error Loading model ", error)
    else:
        print("Model Summary :")
        model.summary()
        scores = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        return scores