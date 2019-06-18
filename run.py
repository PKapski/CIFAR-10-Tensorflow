import numpy as np
import pickle
import os
import download
import keras
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator

cifar_10_dir = 'Dataset/cifar-10-batches-py/'
cifar_10_download_url='Dataset/'
cifar_10_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
save_dir = os.path.join(os.getcwd(), 'saved_models')
log_dir = os.path.join(save_dir, 'logs')
numberOfClaases = 10


def maybe_download_and_extract():
    download.maybe_download_and_extract(url=cifar_10_url, download_dir=cifar_10_download_url)


def getPath(filename=""):
    return os.path.join(cifar_10_dir, filename)


def unpickle(filename):
    with open(filename, mode='rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def normalize(data):
    data = data.astype('float32')
    data /= 255.0
    return data


def one_hot_encode(labels):
    encoded_table = np.zeros((len(labels), numberOfClaases))
    for i, label_value in enumerate(labels):
        encoded_table[i][label_value] = 1
    return encoded_table


def saveModel(save_dir, model_name, model):
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def createModel(train_data):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClaases))
    model.add(Activation('softmax'))
    return model


def createAndTrainModel(_batchSize, _numberOfEpochs, _dataAugmentation, _modelName, train_data, train_labels, test_data,
                        test_labels, learning_rate):
    # Model properties
    batchSize = _batchSize
    numberOfEpochs = _numberOfEpochs
    dataAugmentation = _dataAugmentation
    modelPath = _modelName + '.h5'
    logPath = os.path.join(log_dir, _modelName)
    logPath = logPath + ".log"

    model = createModel(train_data)
    # initiate optimizer
    opt = keras.optimizers.Adam(learning_rate, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    csv_logger = CSVLogger(logPath, separator=',', append=True)
    if not dataAugmentation:
        model.fit(train_data, train_labels,
                  batch_size=batchSize,
                  epochs=numberOfEpochs,
                  validation_data=(test_data, test_labels),
                  shuffle=True,
                  callbacks=[csv_logger])
    else:
        print("Using data augmentation")
        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(train_data)
        model.fit_generator(datagen.flow(train_data, train_labels, batch_size=batchSize),
                            epochs=numberOfEpochs,
                            steps_per_epoch=len(train_data) // batchSize,
                            validation_data=(test_data, test_labels),
                            callbacks=[csv_logger])

    saveModel(save_dir, modelPath, model)


def loadAndEvaluateModel(test_data, test_labels, modelName):
    # Loading model --------------
    model = load_model(os.path.join(save_dir, modelName) + ".h5")
    loss, acc = model.evaluate(test_data, test_labels)

    model.summary()
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def load_cifar():
    maybe_download_and_extract() #Downloads dataset if it doesn't exists
    train_data = None
    train_labels = None
    # Getting label names
    metaDataDict = unpickle(getPath('batches.meta'))
    labelNames = metaDataDict[b'label_names']
    labelNames = np.array(labelNames)
    print("Label names:")
    print(labelNames)
    # Getting train data and labels
    for i in range(1, 6):
        trainDataDict = unpickle(getPath('data_batch_' + str(i)))
        if train_data is None:
            train_data = trainDataDict[b'data']
            train_labels = trainDataDict[b'labels']
        else:
            # Linking data and labels from all batches together
            train_data = np.concatenate((train_data, trainDataDict[b'data']), axis=0)
            train_labels = np.concatenate((train_labels, trainDataDict[b'labels']), axis=0)
    # Trasforming data
    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = train_data.transpose(0, 2, 3, 1)

    testDataDict = unpickle(getPath('test_batch'))
    test_data = testDataDict[b'data']
    test_labels = np.array(testDataDict[b'labels'])
    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = test_data.transpose(0, 2, 3, 1)
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)
    return train_data, train_labels, test_data, test_labels


def main():
    train_data,train_labels,test_data,test_labels = load_cifar()
    trainModel = False
    loadModel = True
    # MODEL------------------------------------
    learningRate = 0.001
    batchSize = 16
    numberOfEpochs = 1
    dataAugmentation = True
    modelName = 'keras_cifar10_trained_model-test' #Without extension

    if trainModel:
        createAndTrainModel(batchSize, numberOfEpochs, dataAugmentation, modelName, train_data, train_labels, test_data,
                            test_labels, learningRate)
    if loadModel:
        loadAndEvaluateModel(test_data, test_labels, modelName)


main()
