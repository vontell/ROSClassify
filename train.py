import numpy as np
import keras
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import h5py

classes = ["coral", "lego", "floor"]
num_classes = len(classes)
batch_size = 16
epochs = 5
image_size = (200, 200, 4) #width, height, channels (r,g,b,hue,entropy?)

# READ IN DATA
root = "training/"
data = {"lego": [], "floor": [], "coral": [], "sand": [], "unknown": []}
for sub, dirs, files in os.walk(root):
    for dir in dirs:
        key = dir
        for subdir, dirs, files in os.walk(root + dir):
            for file in files[0:min(190, len(files))]:
                if file.endswith(".png"):
                    img = Image.open(subdir + "/" + file)
                    Rvals = np.array(img.getdata(band=0)).reshape((200,200))
                    Gvals = np.array(img.getdata(band=1)).reshape((200,200))
                    Bvals = np.array(img.getdata(band=2)).reshape((200,200))
                    hues = np.array(img.convert("HSV").getdata(band=0)).reshape((200,200))

                    # Normalize
                    Rvals = Rvals / 255.0
                    Gvals = Gvals / 255.0
                    Bvals = Bvals / 255.0
                    hues = hues / 360.0

                    RGBHimage = np.stack((Rvals, Gvals, Bvals, hues), axis=-1)
                    data[key].append(RGBHimage)
            print("Finished loading " + key)

total = sum([len(data[arr]) for arr in classes])
for key in classes:
    percent = len(data[key])/total
    print("Loaded has " + str(percent) + " ratio of " + str(key))

x_train, y_train, x_test, y_test = [], [], [], []
train_to_test_ratio = 0.80

# 1. SPLIT DATA INTO TEST AND TRAIN (make sure categories are vectorized)

for i in range(len(classes)):
    clas = classes[i]
    y_vec = keras.utils.to_categorical(classes.index(clas), num_classes)
    for image in data[clas]:
        if np.random.ranf() < train_to_test_ratio:
            x_train.append(image)
            y_train.append(np.copy(y_vec))
        else:
            x_test.append(image)
            y_test.append(y_vec)

print("Created training set of size " + str(len(x_train)))
print("Created testing set of size " + str(len(x_test)))

x_train, y_train = shuffle(x_train, y_train)
print("Shuffled training data")


# 2. CREATE CNN MODEL 

# Build the model, starting with 2 convolutions, max pool, and a dropout
# Our input is an image broken up into 3 (5) channels - R, G, B, (entropy, hue)
model = Sequential()
model.add(Conv2D(32, (10, 10), activation='relu', input_shape=(image_size[0], image_size[1], image_size[2])))
#model.add(Conv2D(32, (10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Repeat the layers from before, but with more filters
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten, dense, dropout, and classify with softmax
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# Compile with the SGD loss (maybe use Adam?)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use ImageDataGenerator to modify images to get more training data. Then fit the model
#   If needed, define preprocessing_function() from https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(rotation_range = 30, featurewise_center=True)
flowed = datagen.flow(np.array(x_train), np.array(y_train), batch_size=batch_size)
model.fit_generator(flowed, steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

# Save the model and evaluate
model.save('grand_challenge_trained_4.h5')
result = model.evaluate(np.asarray(x_test), np.asarray(y_test))
print('\nTesting loss: {}, acc: {}\n'.format(result[0], result[1]))


# Utility functions

def get_entropy_layer():
    pass

def get_hue_layer():
    pass