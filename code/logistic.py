"""
Performs classification using Logistic Regression.
"""

from utils import is_train, process_tweets, save_results_to_csv, predicting_neural_model, train_validate_save_model
from var import *
from keras.models import Sequential, load_model
from keras.layers import Dense


def build_model():
    sequential_model = Sequential()
    sequential_model.add(Dense(1, input_dim=vocab_size, activation="sigmoid"))
    sequential_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return sequential_model


if __name__ == "__main__":
    train = is_train()

    if train:
        model = build_model()
        nb_epochs = 20

        model_file_name = "../models/logistic.keras"

        train_validate_save_model(model, nb_epochs, model_file_name)

    else:
        try:
            model = load_model("../models/logistic.keras")
        except OSError:
            print("First train the model.")
            exit()

        test_tweets = process_tweets(test_processed_file, test_file=True)
        predictions = predicting_neural_model(test_tweets, model)

        save_results_to_csv(predictions, "../results/logistic.csv")
