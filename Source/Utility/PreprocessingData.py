from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.utils import normalize

class PreprocessData:
    def __init__(self,image_directory_train, channel, shuffleData = False, normalData = False, number_of_classes = None ,**kwargs):
        self.image_directory_train = image_directory_train
        self.channel = channel
        self.label_data = None
        self.features_data = None
        self.categorical_label_data = None
        self._number_of_classes = number_of_classes
        self.process(shuffleData,normalData,**kwargs)


    def process(self,shuffleData,normalData, **kwargs):
        try:
            self.load_images_to_data()
            self.categorical_Y()
        except :
            print("Fail to load data from provided directory")
            return

        if shuffleData:
            try:
                self.shuffle_data()
            except:
                print("Fail to shuffle data")

        if normalData:
            try:
                self.normalize_data()
            except:
                print("Fail to normalize data")


    def load_images_to_data(self):
        try:
            list_of_files = os.listdir(self.image_directory_train)
        except:
            print("no file is in the directory")
            raise Exception

        for image_label in list_of_files:
            list_of_label = os.listdir(os.path.join(self.image_directory_train,image_label))
            for file in list_of_label:
                image_file_name = os.path.join(self.image_directory_train, image_label,file)
                if ".png" in image_file_name:
                    img = Image.open(image_file_name).convert("L")
                    width,height = img.size
                    img = np.resize(img, (width,height,self.channel))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(1,width,height,self.channel)

                if self.features_data is not None and self.label_data is not None:
                    self.features_data = np.append(self.features_data, im2arr, axis=0)
                    self.label_data = np.append(self.label_data,[int(image_label)],axis=0)
                else:
                    if any([self.features_data,self.label_data]):
                        raise ValueError("data of both feature and label should be None")
                    self.features_data = im2arr
                    self.label_data = [int(image_label)]

    def shuffle_data(self):
        self.features_data, self.label_data = shuffle(self.features_data, self.label_data)

    def normalize_data(self):
        self.features_data = normalize(self.features_data, axis=1)

    @property
    def number_of_classes(self):
        if self._number_of_classes:
            return self._number_of_classes
        elif any(self.label_data):
            return len(set(self.label_data))
        else:
            return None

    def categorical_Y(self):
        if self.number_of_classes:
            self.categorical_label_data= np_utils.to_categorical(self.label_data, self.number_of_classes)

