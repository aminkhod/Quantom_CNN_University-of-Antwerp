from __future__ import division, print_function, absolute_import
from glob import glob
from scipy.misc import imresize
from skimage import  color, io
import numpy as np

import os


files_pathGTI ='../ground truth images'
files_pathNI = '../noisy images'

files_pathValGTI ='../Validation ground truth images'
files_pathValNI = '../Validation noisy images'

pic_pathGTI = os.path.join(files_pathGTI, '*.jpg')
pic_pathNI = os.path.join(files_pathNI, '*.jpg')
pic_pathValGTI = os.path.join(files_pathValGTI, '*.jpg')
pic_pathValNI = os.path.join(files_pathValNI, '*.jpg')

Sort_pic_pathGTI = sorted(glob(pic_pathGTI))
Sort_pic_pathNI = sorted(glob(pic_pathNI))
Sort_pic_pathValGTI = sorted(glob(pic_pathValGTI))
Sort_pic_pathValNI = sorted(glob(pic_pathValNI))


n_files = len(Sort_pic_pathGTI) + len(Sort_pic_pathNI) + len(Sort_pic_pathValGTI) +len(Sort_pic_pathValNI)
print(n_files)

size_image = 64

y_train = np.zeros((len(Sort_pic_pathGTI), size_image, size_image), dtype='float64')

count = 0
for f in Sort_pic_pathGTI:
    try:
        img = io.imread(f)
        gray_img = color.rgb2gray
        new_img = imresize(gray_img, (size_image, size_image))
        y_train[count] = np.array(new_img)
        count += 1
    except:
        continue

x_train = np.zeros((len(Sort_pic_pathNI), size_image, size_image), dtype='float64')
count = 0
for f in Sort_pic_pathNI:
    try:
        img = io.imread(f)
        gray_img = color.rgb2gray
        new_img = imresize(gray_img, (size_image, size_image))
        x_train[count] = np.array(new_img)
        count += 1
    except:
        continue

y_validation = np.zeros((len(Sort_pic_pathValGTI), size_image, size_image), dtype='float64')
count = 0
for f in Sort_pic_pathValGTI:
    try:
        img = io.imread(f)
        gray_img = color.rgb2gray
        y_validation[count] = np.array(new_img)
        count += 1
    except:
        continue

x_validation = np.zeros((len(Sort_pic_pathValNI), size_image, size_image), dtype='float64')
count = 0
for f in Sort_pic_pathValNI:
    try:
        img = io.imread(f)
        gray_img = color.rgb2gray
        new_img = imresize(gray_img, (size_image, size_image))
        x_validation[count] = np.array(new_img)
        count += 1
    except:
        continue



from keras.layers import Input, Conv2D
from keras.models import Model


input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format

x1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)

x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x1)

x3 = Conv2D(64, (5, 5), activation='relu', padding='same')(x2)

x4 = Conv2D(64, (5, 5), activation='relu', padding='same')(x3)

x5 = Conv2D(64, (5, 5), activation='relu', padding='same')(x4)

x6 = Conv2D(64, (5, 5), activation='relu', padding='same')(x5)

x7 = Conv2D(64, (5, 5), activation='relu', padding='same')(x6)

x8 = Conv2D(64, (5, 5), activation='relu', padding='same')(x7)

x9 = Conv2D(64, (5, 5), activation='relu', padding='same')(x8)

x10 = Conv2D(64, (5, 5), activation='relu', padding='same')(x9)

x11 = Conv2D(64, (5, 5), activation='relu', padding='same')(x10)

output = Conv2D(1, (5, 5), activation='relu')(x11)


CNN_Ant = Model(input_img, output)
CNN_Ant.compile(optimizer='adam', loss='mean_absolute_error')
CNN_Ant.summary()
CNN_Ant.fit(x_train, y_train,
                epochs=15000,
                batch_size=10000,
                shuffle=True,
                validation_data=(x_validation, y_validation))

CNN_Ant_imgs = CNN_Ant.predict(x_validation)

#saviny model for another use
CNN_Ant.save("Denoising convolutional neural network model", overwrite=False, include_optimizer=True)


#################################
####### Visualisation ###########
#################################

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many pictures we will display
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(y_validation[i])
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(CNN_Ant_imgs[i])
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
