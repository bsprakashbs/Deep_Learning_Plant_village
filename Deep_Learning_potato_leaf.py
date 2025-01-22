import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.layers import Resizing, RandomFlip, RandomRotation ,Rescaling
import matplotlib.pyplot as plt


#-----------------------------------------Load Image using TEnsor flow ----------------------------------------#
#256 ----> 0 to 255 RGB scale
IMAGE_SIZE=256
BATCH_SIZE=32
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
#----------------------------------------Data Augmentation  -- pick same iamge train that in to  contrast,horizontal flip,rotation,zoom etc....----------------#
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2)
])