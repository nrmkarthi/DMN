import tensorflow as tf
from tensorflow.keras import layers, models
from maxout_layer import Maxout

class DMNWO(models.Model):
    def __init__(self):
        super(DMNWO, self).__init__()

        # Define convolutional layers
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')

        # Define maxout layers
        self.maxout1 = Maxout(16)
        self.maxout2 = Maxout(32)
        self.maxout3 = Maxout(64)

        # Define other layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense3 = layers.Dense(1, activation='sigmoid')

        # Define batch normalization layers
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.batch_norm3 = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.maxout1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.maxout2(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x, training=training)
        x = self.maxout3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)
