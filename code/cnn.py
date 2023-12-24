"""
Performs classification using CNN.
"""
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Conv1D
from utils import is_train, predicting_cnn_model, initialize_glove
from var import *


if __name__ == "__main__":
    max_length = 40

    train = is_train()

    if train:
        dim = 200

        tweets, labels, embedding_matrix = initialize_glove(dim, max_length)

        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
        model.add(Dropout(0.4))
        model.add(Conv1D(600, 3, padding="valid", activation="relu", strides=1))
        model.add(Conv1D(300, 3, padding="valid", activation="relu", strides=1))
        model.add(Conv1D(150, 3, padding="valid", activation="relu", strides=1))
        model.add(Conv1D(75, 3, padding="valid", activation="relu", strides=1))
        model.add(Flatten())
        model.add(Dense(600))
        model.add(Dropout(0.5))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(tweets, labels, batch_size=128, epochs=8, validation_split=0.1, shuffle=True)

        model.save("../models/cnn.keras")
        print(f"\nSaved to ../models/cnn.keras")

    else:
        try:
            model = load_model("../models/cnn.keras")
        except OSError:
            print("First train the model.")
            exit()

        output_file = "cnn.csv"
        predicting_cnn_model(model, max_length, output_file)
