import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image
from skimage.transform import resize
from random import shuffle
from random import randint
import math
import random
from io import BytesIO
import jpeg4py as jpeg
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

list_paths = []
for subdir, dirs, files in os.walk("/home/gasimov_aydin/ieee"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)
        
list_train = [filepath for filepath in list_paths if "train_new/" in filepath]
shuffle(list_train)
list_test = [filepath for filepath in list_paths if "test/" in filepath]

list_train = list_train
list_test = list_test
index = [os.path.basename(filepath) for filepath in list_test]

list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_paths if "train" in filepath]))

list_classes = ['Sony-NEX-7',
 'Motorola-X',
 'HTC-1-M7',
 'Samsung-Galaxy-Note3',
 'Motorola-Droid-Maxx',
 'iPhone-4s',
 'iPhone-6',
 'LG-Nexus-5x',
 'Samsung-Galaxy-S4',
 'Motorola-Nexus-6']
 
img_size_1 = 512

img_size_2 = 512
 
def get_class_from_path(filepath):
    return os.path.dirname(filepath).split(os.sep)[-1]

MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

def random_manipulation(img, manipulation=None):

    if 1 == 1:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded

def crop(img):
    width, height = img.size  # Get dimensions

    left = (width - 128) / 2
    top = (height - 128) / 2
    right = (width + 128) / 2
    bottom = (height + 128) / 2
    #center = randint(300, 1200)
    #left = center - 299
    #top = center -299
    #right = center + 299
    #bottom = center + 299

    return np.array(img.crop((left, top, right, bottom)))

def get_crop(img, crop_size, random_crop=True):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]

def read_and_resize(filepath):
    #im_array = np.array(Image.open(filepath).convert('RGB'), dtype="uint8")
    im_array = np.array(Image.open(filepath), dtype="uint8")
    #pil_im = Image.fromarray(im_array)
    #im_array = np.array(cv2.imread(filepath))
    #new_img = pil_im.resize(( , img_size_2))
    #w, h = pil_im.size
    #w, h = pil_im.size
    #img = pil_im.crop((w // 2 - 128, h // 2 - 128, w // 2 + 128, h // 2 + 128))
    #new_array = np.array(crop(pil_im))
    #print(new_array.shape)
    #img = crop_img(pil_im,0,np.random.randint(0, 5))
    #img_result = []
    #for i in range(0,10):
    #   img_result.append(crop(pil_im))
    new_array = np.array(get_crop(im_array ,512,True))
    #new_array = np.array(img.resize((img_size_1 , img_size_2)))
    #new_array = np.array(img) 
    return new_array

def read_and_resize_test(filepath):
    manip = 0
    #print(filepath)
    if ('_manip.tif' in filepath):
       manip = 1
       #print(1)
    else:
       manip = 0
       #print(0)

    #im_array = np.array(Image.open(filepath).convert('RGB'), dtype="uint8")
    im_array = np.array(Image.open(filepath), dtype="uint8")
    #pil_im = Image.fromarray(im_array)
    #im_array = np.array(cv2.imread(filepath))
    #w, h = pil_im.size
    #new_img = pil_im.resize(( , img_size_2))
    #w, h = pil_im.size
    if (manip == 1):
       img2 = random_manipulation(im_array)
       new_array = np.array(get_crop(img2,512,True))
    else:
       new_array = im_array #np.array(get_crop(im_array,224,True))
    #print(new_array.shape)
    #img2 = img2.crop((w // 2 - 128, h // 2 - 128, w // 2 + 128, h // 2 + 128))
    #img = crop_img(pil_im,0,np.random.randint(0, 5))
    #new_array = np.array(pil_im.resize((img_size_1 , img_size_2)))
    #new_array = np.array(get_crop(img2,128,True))
    #new_array = np.array(img)
    #print(new_array.shape)
    return new_array

def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    return labels, label_index 
    
#X_train = np.array([read_and_resize(filepath) for filepath in list_train])
#X_test = np.array([read_and_resize_test(filepath) for filepath in list_test])

X_train_data = []
X_test_data = []
X_labels = []


for i in range(0,len(list_train)-14330):
    #print(list_train[i],' ', len(list_train),' - ', y)
    xu = read_and_resize(list_train[i])
    if (xu.shape == (512,512,3)):
       X_train_data.append(xu)
       X_labels.append(get_class_from_path(list_train[i]))
       #print(xu.shape, '-', get_class_from_path(list_train[i]))
    #print(np.array(X_train_data).shape, '-', len(list_train))

X_train = np.array(X_train_data)
print('Train shape: ', X_train.shape)

for i in range(0,len(list_test)-2630):
    X_test_data.append(read_and_resize_test(str(list_test[i])))
    #print(list_test[i],' ', len(list_test),' file: ', filepath)
    #print(np.array(X_test_data).shape)

X_test = np.array(X_test_data)
print('Test shape: ', X_test.shape)

#labels = [get_class_from_path(filepath) for filepath in list_train]
y, label_index = label_transform(X_labels)
y = np.array(y)

from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.optimizers import Adam

inp = Input(shape=(img_size_1, img_size_2, 3), name="X_1")  #(img_size_1, img_size_2, 3)
nclass = len(label_index)

def get_model():

    base_model = InceptionResNetV2(include_top=False, weights='imagenet' ,
                 input_tensor=inp, classes=nclass)
    x = base_model.output
 
    x1 = GlobalMaxPooling2D()(x)
    merge_one = Dense(1024, activation='relu', name='fc2')(x1)
    merge_one = Dropout(0.5)(merge_one)
    merge_one = Dense(256, activation='relu', name='fc3')(merge_one)
    merge_one = Dropout(0.5)(merge_one)
        
    predictions = Dense(nclass, activation='softmax')(merge_one)
    
    model = Model(input=base_model.input, output=predictions)
   
    #model.load_weights('weightsV2.best.hdf5')
 
    sgd = Adam(lr=1e-4, decay=1e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model 
    
model = get_model()
#print('Train shape: ',X_train.shape)
file_path="weightsV2.best.hdf5"
model.summary()

model.load_weights('weightsV2.best.hdf5')

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')

early = EarlyStopping(monitor="val_loss", patience=15)

callbacks_list = [checkpoint, early] #early

model.fit(X_train, y, validation_split=0.1, epochs=100, shuffle=True, verbose=1, batch_size = 22,
                              callbacks=callbacks_list)

#print(history)
#model.save_weights('my_model_weights2.h5')

#model.load_weights(file_path)

predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
predicts = [label_index[p] for p in predicts]

df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = index
df['camera'] = predicts
df.to_csv("subV2.csv", index=False)
