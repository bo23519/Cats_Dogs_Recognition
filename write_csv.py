import csv
import os
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from numpy import asarray
from numpy import save

generate = True

rawDataset = 'trainTest/test/'
prediction = list()
model = load_model('cats_and_dogs_small_3.h5')
cnt = 1

if generate:
    for i in os.listdir(rawDataset):
        img = load_img(rawDataset + i, target_size=(150, 150))
        img = img_to_array(img)
        img = img.reshape(1, 150, 150, 3)
        img = img.astype('float32')
        pre = int(model.predict(img)[0][0])
        print(pre)
        prediction.append(pre)

with open('sample_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow (["id", "label"])
    for pre in prediction:
        writer.writerow([cnt, pre])
        print(pre)
        cnt = cnt + 1

# img = load_img(rawDataset+'dog.1300.jpg', target_size=(150, 150))
# img = img_to_array(img)
# img = img.reshape(1, 150, 150, 3)
# # img = img.astype('float32')
# # print(img)
# pre = model.predict(img)
# num = pre[0][0]
# print(round(num))

