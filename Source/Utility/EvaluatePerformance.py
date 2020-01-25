# Evaluate the model performance
# Future Override
def EvaluatePerformance(model, x_test, y_test):
    test_acc: object
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
    return test_acc