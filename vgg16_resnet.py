from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2

train_dir = 'trainTest/train'
validation_dir = 'trainTest/validation'

# model = VGG16(include_top=False, input_shape=(150,150,3))  # Using this line to train VGG16
model = ResNet50V2(include_top=False, input_shape=(150,150,3))  # Using this line to train ResNet
for lay in model.layers:
    lay.trainable = False  # set to not trainable

# Fine-tuning, add classifier to the top
flat = layers.Flatten()(model.layers[-1].output)
dense = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat)
output = layers.Dense(1, activation='sigmoid')(dense)
model = models.Model(inputs=model.inputs, outputs=output)

# Compile model
opt = optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=30,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=30,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_3.h5')

# Print graph for accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = len(acc)

epochs = range(1, epoch + 1)
plt.plot(epochs, acc, 'k', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy % ')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'k', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss % ')
plt.title('Training and validation loss')
plt.legend()
plt.show()
