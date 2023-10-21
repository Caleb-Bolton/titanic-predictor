import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from helper import *
from SimpleNN import SimpleNN
import numpy as np
import matplotlib.pyplot as plt

X, y = preprocess_titanic_dataset('train.csv')
X_train, y_train, X_val, y_val, X_test, y_test = split_training_and_validation_sets(X, y)

# mps_device = torch.device("mps")
# model.to(mps_device)

def test_models(X_train, y_train, X_val, y_val, n_epochs):
    input_dim = X_train.shape[1]
    output_dim = 1
    models = [
        SimpleNN([input_dim, 5, 3, output_dim]),
        SimpleNN([input_dim, 15, 10, 1, output_dim]),
        SimpleNN([input_dim, 50, 25, 15, 10, output_dim]),
        SimpleNN([input_dim, 50, 50, 50, 50, 50, output_dim]),
        # SimpleNN([input_dim, 100, 50, 50, 25, 25, 10, 10, 5, output_dim])
    ]

    overall_max_accuracy_val = 0
    overall_max_accuracy_val_model = -1

    for i in range(len(models)):
        model = models[i]
        print(f'Training {model}')
        accuracy_train_list, accuracy_val_list = train_neural_net(model, n_epochs, X_train, y_train, X_val, y_val, 'model.pth')
        max_accuracy_val_index = accuracy_val_list.index(max(accuracy_val_list))
        if accuracy_val_list[max_accuracy_val_index] > overall_max_accuracy_val:
            overall_max_accuracy_val = accuracy_val_list[max_accuracy_val_index]
            overall_max_accuracy_val_model = i
        print(f'Max validation accuracy: {accuracy_val_list[max_accuracy_val_index]}')
        plt.figure(i + 1)
        plt.plot(accuracy_train_list, label='Training Accuracy')
        plt.plot(accuracy_val_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show(block=False)
    print(f'Best model: {models[overall_max_accuracy_val_model]}')
    print(f'Accuracy: {overall_max_accuracy_val}')
    plt.show()


test_models(X_train, y_train, X_val, y_val, 5000)
input()



# # Run the model on the cvs data
# with torch.no_grad():
#     input_dim = X_val.shape[1]
#     output_dim = 1
#     num_layers = 3
#     num_units = 3

#     model = SimpleNN(input_dim, output_dim, num_layers, num_units)
#     model.load_state_dict(torch.load('model.pth'))  # Load the trained model state
#     model.eval()  # Set the model to evaluation mode

#     y_pred = model(X_test)

# # Convert predictions to NumPy and process as needed
# y_pred = y_pred.numpy()

# # Now `y_pred` contains the model's predictions on the test set
# y_pred_rounded = np.round(y_pred)
# correct_predictions = y_pred_rounded == y_val.numpy()

# # Calculate accuracy
# accuracy = np.mean(correct_predictions.astype(int))  # Convert boolean to int and then take mean

# print(f'Accuracy: {accuracy * 100}%')