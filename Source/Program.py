import matplotlib.pyplot as plt
import Source.Utility.PreprocessingData as pre
from Source.BuildModel.TrainKerasModel import TrainKerasModel
from Source.Predict.ModelPredict import predictData
from Source.Utility.EvaluatePerformance import EvaluatePerformance

if __name__ == "__main__":

    # Load Training data
    image_directory_train = r'C:\Users\Mengmeng\PycharmProjects\Tilter\DigitRecognition\Data\mnist\training'
    #image_directory_train = r'C:\Users\Mengmeng\PycharmProjects\Tilter\DigitRecognition\Data\mnist\trainingSmall'
    trainData = pre.PreprocessData(image_directory_train,channel=1,shuffleData=True,normalData=True)
    x_train = trainData.features_data
    y_train = trainData.label_data


    # Pass transfer model if any and train model
    trainedmodel = TrainKerasModel(x_train, y_train)


    # Make Prediction
    image_directory_test = r'C:\Users\Mengmeng\PycharmProjects\Tilter\DigitRecognition\Data\mnist\testing'
    testData = pre.PreprocessData(image_directory_test,channel=1,shuffleData=False,normalData=True)
    x_test = testData.features_data
    y_test = testData.label_data
    prediction = predictData(trainedmodel.model,x_test)
    print(prediction[0])
    plt.imshow(x_test[0].reshape(28, 28), cmap="Greys")


    # Evaluate the model performance
    test_acc = EvaluatePerformance(trainedmodel.model, x_test, y_test)
    # Print out the model accuracy
    print('\nTest accuracy:', test_acc)
