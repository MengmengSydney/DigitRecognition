# Future Override
def predictData(model, x_test):
    predictions = model.predict_classes([x_test])
    return predictions