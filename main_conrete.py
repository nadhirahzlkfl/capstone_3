#%% import packages
import os,keras,cv2
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime
from keras import layers,losses,metrics,callbacks,applications,optimizers

#%% data loading
data_path=os.path.join(os.getcwd(),'dataset','Concrete Crack Images for Classification')

BATCH_SIZE=10
IMG_SIZE=(160,160)


train_dataset, validation_dataset = keras.utils.image_dataset_from_directory(data_path,
                                                          shuffle=True,
                                                          batch_size=BATCH_SIZE,
                                                          image_size=IMG_SIZE,
                                                          validation_split=0.3,
                                                          subset='both',
                                                          seed=69)

#%% inspect some data samples from dataset
class_names=train_dataset.class_names
print(class_names)
batch_1=train_dataset.take(1)
for feature,label in batch_1:
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(feature[i].numpy().astype('uint8'))
        plt.title(class_names[label[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
plt.show()

#%% split validation dataset into 2 equal splits: for val and test
nBatches=validation_dataset.cardinality().numpy()
print(nBatches)
val_dataset=validation_dataset.take(nBatches//2)
test_dataset=validation_dataset.skip(nBatches//2)

# convert the val and test dataset back to PrefetchDataset
val_dataset=val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # preload the data into memory
test_dataset=test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#%% data augmentation
# create Sequential model for augmentation
augmentation=keras.Sequential()
augmentation.add(layers.RandomFlip())
augmentation.add(layers.RandomRotation(factor=0.2))
feature_augmented=augmentation(feature)

#%% inspect feature augmented
class_names=train_dataset.class_names
print(class_names)
batch_1=train_dataset.take(1)
for feature,label in batch_1:
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(feature_augmented[i].numpy().astype('uint8'))
        plt.title(class_names[label[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
plt.show()

#%% create preprocessing layer
preprocess_input=applications.mobilenet_v2.preprocess_input

#%% transfer learning
# load in the pretrained model as feature extractor
IMG_SHAPE=IMG_SIZE+(3,)
base_model=applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# freeze the entire feature extractor
base_model.trainable=False
base_model.summary()

# construct the classifier
global_avg=layers.GlobalAveragePooling2D()
output_layer=layers.Dense(len(class_names),activation='softmax')

# use functional API to connect all the layers together
# begin with the input
inputs=keras.Input(shape=IMG_SHAPE)
# augmentation layers
x=augmentation(inputs)
# preprocessing layer
x=preprocess_input(x)
# feature extractor
x=base_model(x)
# classifier
x=global_avg(x)
outputs=output_layer(x)
# create model
model=keras.Model(inputs=inputs,outputs=outputs)
model.summary()
keras.utils.plot_model(model)

#%% model compilation
optimizer=optimizers.Adam(learning_rate=0.00001)
loss=losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# tensorboard and earlystopping callbacks
logpath=os.path.join('transfer_learning_log',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
print(logpath)
tb=callbacks.TensorBoard(log_dir=logpath)
es=callbacks.EarlyStopping(patience=1,verbose=1)

#%% model training
epochs=10
history_first=model.fit(train_dataset,validation_data=val_dataset,epochs=epochs,callbacks=[tb,es])

#%% model evaluation
model.evaluate(test_dataset)

#%% use the model to make prediction
for image_batch, label_batch in val_dataset.take(1):
    y_pred = np.argmax(model.predict(image_batch),axis=1)
    predicted_class = [class_names[x] for x in y_pred]
print(predicted_class)

plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    plt.title(f'prediction label: {predicted_class[i]}, label: {class_names[label_batch[i]]}')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
plt.show()

#%% use the model to predict a sample image
filepath=os.path.join(os.getcwd(),'static','sample.jpg')
img=cv2.imread(filename=filepath,flags=cv2.IMREAD_COLOR_RGB)
img=cv2.resize(img,dsize=(160,160))
plt.imshow(img)
plt.show()
print(img.shape)
img=np.expand_dims(img,axis=0)
print(img.shape)

prediction=model.predict(img)
class_names[np.argmax(prediction,axis=1)[0]]

#%% save model
saved_model=os.path.join(os.getcwd(),'saved_model')
model.save(os.path.join(saved_model,'model.keras'))

# %%
