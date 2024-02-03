from GUI import GUI
from preprocessing import preprocessing, data_extraction_and_splitting
from Multilayer_Perceptron import Multilayer
from Evaluating import Evaluator

if __name__ == "__main__":
    # Data Preprocessing and splitting
    data_set, scaler = preprocessing()
    x_train, x_test, y_train, y_test = data_extraction_and_splitting(data_set)

    # Getting the hyperparameters
    hidden_neurons, learning_rate, epochs_num, use_bias, activation_fn = GUI().Inputs

    # Initializing the model
    model = Multilayer(activation_fn, learning_rate, epochs_num, use_bias, hidden_neurons)

    # Training & Testing the Multilayer model
    predicted_train = model.train(x_train, y_train)
    predicted_test = model.test(x_test)

    # Evaluating:
    Evaluator(model, scaler, y_train, y_test, predicted_train, predicted_test)
