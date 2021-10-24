import tensorflow as tf
import numpy as np
import os
from IPython import embed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from functools import partial
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import sys
import cv2

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16


tfrecord_path = "/home/deploy/shubham/pratice/classification/train.record"
CLASSES = 67
IMAGE_SIZE = [120, 120]



print(sys.argv)


def decode_image(image):
	# image = tf.image.decode_jpeg(image, channels = 3)
	print(image)
	# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image = tf.io.decode_jpeg(image)
	image = tf.cast(image, tf.float32)
	image = tf.compat.v1.image.resize(image, [120,120])
	return image

def read_tfrecord(example):
	tfrecord_format = ({
		"image": tf.io.FixedLenFeature([], tf.string),
		"label": tf.io.FixedLenFeature([], tf.int64)
		})
	example = tf.io.parse_single_example(example, tfrecord_format)
	x_train = decode_image(example["image"])
	label = example["label"]
	# label = tf.nn.embedding_lookup(np.identity(10), label)
	label = tf.one_hot(label, depth = CLASSES)
	return x_train,label

def load_dataset(filename):
	ignore_order = tf.data.Options()
	ignore_order.experimental_deterministic = False
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.with_options(ignore_order)
	dataset = dataset.map(partial(read_tfrecord))
	return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(10)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

train  = get_dataset(tfrecord_path)
image_train, label_train = next(iter(train))


initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps = 20, decay_rate = 0.95, staircase = True)




def make_model():
    base_model = tf.keras.applications.Xception(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(67, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

model = make_model()

history = model.fit(image_train, label_train,epochs=20, verbose = 1)


print("#####################################")
print("#####################################")
print(model.summary())
print("#####################################")
print("#####################################")


### Evaluating on the train dataset
model.evaluate(image_train,label_train)

# Checking the graph for the accuracy and the loss throughout the training
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
