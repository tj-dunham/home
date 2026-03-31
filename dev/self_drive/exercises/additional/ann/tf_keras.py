import tensorflow as tf
import matplotlib.pyplot as plt

# set of numbers for OCR
mnist = tf.keras.datasets.mnist

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# need to normalize the data
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

# Sequential for feed forward - most common
model = tf.keras.models.Sequential()
# Layer 1 Flatten input layer to condition data
model.add(tf.keras.layers.Flatten())
# Layer 2 dense hidden layer
# Values - number of neurons = 128, activation = tf.nn.relu
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
# Layer 3 Output layer, 10 neurons, one for each potential digit, activation is softmax to get stats
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

# Now params for training of model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)

model.save('epic_num_reader.keras')

new_model = tf.keras.models.load_model('epic_num_reader.keras')
predictions = new_model.predict([x_test])

# will print one hot arrays with probabilities
print(predictions)

import numpy as np
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()


#plt.imshow(x_train[0] cmap = plt.cm.binary)
#plt.show()

#print(x_train[0])
