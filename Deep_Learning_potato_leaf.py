import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.layers import Resizing, RandomFlip, RandomRotation ,Rescaling
import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------Load Image using TEnsor flow ----------------------------------------#
#256 ----> 0 to 255 RGB scale
TF_ENABLE_ONEDNN_OPTS=0
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE )
class_names=dataset.class_names
print(class_names)

#------------------------------Image stored in 3D array RGB. print the image to check---------------------------#
plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    for i in range(3):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.title(class_names[label_batch[0]])
        plt.axis("off")
        #plt.show()
        print(image_batch[0].shape)
    #print(label_batch.numpy())
print(len(dataset))
#----------------------------------------- 80% --> training----------------------------------------------------------#
#-----------------------------------------20% --> 10% validation ,10% testing ----------------------------------------#
#-----------------------------------------2get_dataset_partitions_tf----------------------------------------#
train_size=0.8
val_size=0.1
# check the split
print(int(len(dataset)*train_size))
print(int(len(dataset)*val_size))
print(int(len(dataset)*val_size))
def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size= len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=12)
    train_size= int(train_split*ds_size)
    val_size = int(val_split * ds_size)
    train_ds=ds.take(train_size)                  # 80 percent data
    val_ds  = ds.skip(train_size).take(val_size) # 10  percent data
    test_ds = ds.skip(train_size).skip(val_size) # 10  percent data
    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))
#----------------------------------------performance tuning- to use cache value-----------------#
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
#----------------------------------------Rescaling and Resizng for model requirements----------------#
#resize_and_rescale  = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),layers.experimental.preprocessing.Rescaling(1.0/255)])
resize_and_rescale = tf.keras.Sequential([
    Resizing(IMAGE_SIZE,IMAGE_SIZE),
    Rescaling(1.0/255)
])
#----------------------------------------Data Augmentation  -- pick same image train that in to  contrast,horizontal flip,rotation,zoom etc....----------------#
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2)
])
#----------------------------------------conventional neural network.- Convolutional and pooling layer...----------------#
input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes =3
model = models.Sequential([
                            resize_and_rescale,
                            data_augmentation,
                            layers.Conv2D(32,(3,3),activation='relu',input_shape =input_shape),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,kernel_size =(3,3) ,activation='relu'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,kernel_size =(3,3) ,activation='relu'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,(3,3) ,activation='relu'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,(3,3) ,activation='relu'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,(3,3) ,activation='relu'),
                            layers.MaxPooling2D((2,2)),
                            layers.Flatten(),
                            layers.Dense(64,activation='relu'),
                            layers.Dense(n_classes, activation='softmax'),
                            ])
model.build(input_shape=input_shape)

print(model.summary())

# compiling  the model

model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
             )

######  Checking how model accuracy is getting improved by giving epochs=50  - 50 times it re fit the model by passing different data set and provides the accuracy
###### cpu - will consume lot
###### GPU -  it takes less time.

EPOCHS = 20
history = model.fit(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    validation_data=val_ds,
                    verbose=1,
                    epochs=EPOCHS,
                    )

print(history)
print(history.params)
print(history.history.keys())
print(type(history.history['loss']))
print(len(history.history['loss']))
print(len(history.history['accuracy']))
scores = model.evaluate(test_ds)
print(scores)

#---------------------------------------- plotting  to see how accuracy is increased..----------------#

print('-----print(acc)-----------')
print(np.array(history.history['accuracy']))
print('-----print(val_acc)-----------')
print(np.array(history.history['val_accuracy']))
#
acc = np.array(history.history['accuracy'])
print('-----print(acc)-----------')
print(acc)
#
val_acc = np.array(history.history['val_accuracy'])
print('---------print(val_acc)--------')
print(val_acc)
#
print('---------length of plotting acc and val_acc values--------')
print(f"acc: {acc}, len: {len(acc)}")
print(f"val_acc: {val_acc}, len: {len(val_acc)}")
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
plt.grid(True)
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.figure(figsize=(8, 8))
print('---------worked 1--------')
plt.subplot(1, 2, 1)
print('---------worked 1.1--------')
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
#plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
#plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')

print('---------worked 1.2--------')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
print('---------worked 2--------')
plt.subplot(1, 2, 2)
#plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(EPOCHS), loss, label='Training Loss')
#plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#---------------------------------------- Run prediction on a sample single image..----------------#
'''
for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names[np.argmax(batch_prediction[0])])
'''

# ---------------------------------------- Run prediction on a multipel images ..----------------#
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = class_names[labels[i]]
            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
            plt.axis("off")
            plt.show()