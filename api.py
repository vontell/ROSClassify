import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

num_classes = 4   #lego, coral, seaweed, sand
batch_size = 32
epochs = 10
image_size = (48, 48, 5) #width, height, channels (r,g,b,entropy,hue?)

# READ IN DATA
# data = ...


x_train, y_train, x_test, y_test = [], [], [], []

# 1. A) SPLIT DATA INTO TEST AND TRAIN (make sure categories are vectorized)
#       Categories are as follows:
#           1 - lego
#           2 - coral
#           3 - seaweed
#           4 - sand



# 2. CREATE CNN MODEL 

# Build the model, starting with 2 convolutions, max pool, and a dropout
# Our input is an image broken up into 5 channels - R, G, B, entropy, hue
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape=(image_size[0], image_size[1], image_size[2])))
model.add(Conv2D(32, (3, 3), activation='relu'))
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
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile with the SGD loss (maybe use Adam?)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Use ImageDataGenerator to modify images to get more training data. Then fit the model
#   If needed, define preprocessing_function() from https://keras.io/preprocessing/image/
datagen = ImageDataGenerator()
flowed = datagen.flow(np.array(x_train), np.array(y_train), batch_size=batch_size)
model.fit_generator(flowed, steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

# Save the model and evaluate
model.save('grand_challenge_trained.h5')
result = model.evaluate(np.asarray(x_test), np.asarray(y_test))
print('\nTesting loss: {}, acc: {}\n'.format(result[0], result[1]))

