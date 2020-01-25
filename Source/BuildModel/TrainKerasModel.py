# future override to add transfer model functionality
import tensorflow as tf

class TrainKerasModel:

    def __init__(self, x_train, y_train , model = None):
        self.x_train = x_train
        self.y_train = y_train
        self.model = tf.keras.models.Sequential()
        self.buildModel()
        self.compileModel()
        self.fitModel()

    #Build the model object
    def buildModel(self):
        # Add the Flatten Layer
        self.model.add(tf.keras.layers.Flatten())
        # Build the input layers
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # Build the hidden layers
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # Build the output layer
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # Compile the model
    def compileModel(self):
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # fit the model
    def fitModel(self):
        self.model.fit(x=self.x_train, y=self.y_train, epochs=5)
