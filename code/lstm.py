"""
Performs classification using LSTM network.
"""
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from utils import is_train, initialize_glove, predicting_cnn_model
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
        model.add(LSTM(128))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(tweets, labels, batch_size=128, epochs=5, validation_split=0.1, shuffle=True)

        model.save("../models/lstm.keras")
        print(f"\nSaved to ../models/lstm.keras")

    else:
        try:
            model = load_model("../models/lstm.keras")
        except OSError:
            print("First train the model.")
            exit()

        output_file = "lstm.csv"
        predicting_cnn_model(model, max_length, output_file)
