"""
Extracts dense vector features from penultimate layer of CNN model and perform SVM classifications on those features.
"""
import pickle
import numpy as np
from numpy import loadtxt
from sklearn import svm
from keras.models import load_model, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import is_train, process_tweets_neural, save_results_to_csv, open_model
from var import *

if __name__ == "__main__":
    train_features_file = "../models/cnn-feats-svm/train-feats.npy"
    train_labels_file = "../models/cnn-feats-svm/train-labels.txt"
    test_features_file = "../models/cnn-feats-svm/test-feats.npy"
    max_length = 40

    train = is_train()

    if train:
        tweets, labels = process_tweets_neural(train_processed_file, test_file=False)
        tweets = pad_sequences(tweets, maxlen=max_length, padding="post")
        shuffled_indices = np.random.permutation(tweets.shape[0])
        tweets = tweets[shuffled_indices]
        labels = labels[shuffled_indices]
        model = load_model("../models/cnn.keras")
        model = Model(model.layers[0].input, model.layers[-3].output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.save("../models/cnn-feats-svm/cnn-feats-svm.keras")
        predictions = model.predict(tweets, batch_size=1024, verbose=1)
        np.save(train_features_file, predictions)
        np.savetxt(train_labels_file, labels)

        x_train = np.load(train_features_file)
        y_train = loadtxt(train_labels_file, dtype=float).astype(int)
        x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

        print("\nExtracting features & training model\n")
        model = svm.LinearSVC(C=1, max_iter=10000, dual="auto")
        model.fit(x_train, y_train)

        print("\nTraining Complete")
        del x_train
        del y_train

        print("\nValidating model")
        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        print("\nVal Accuracy: %.4f %%" % (accuracy * 100.0))

        with open("../models/cnn-feats-svm/cnn-feats-svm.pkl", "wb") as file:
            pickle.dump(model, file)
            file.close()
        print(f"\nSaved to ../models/cnn-feats-svm/cnn-feats-svm.pkl")

    else:
        try:
            model = load_model("../models/cnn-feats-svm/cnn-feats-svm.keras")
        except OSError:
            print("First train the model.")
            exit()
        test_tweets0, _ = process_tweets_neural(test_processed_file, test_file=True)
        test_tweets = []
        for i in test_tweets0:
            feature_vector = i[1]
            test_tweets.append(feature_vector)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding="post")
        predictions = model.predict(test_tweets, batch_size=1024, verbose=1)
        np.save(test_features_file, predictions)

        model = open_model("../models/cnn-feats-svm/cnn-feats-svm.pkl")

        X_test = np.load(test_features_file)
        predictions = model.predict(X_test)
        predictions = predictions.tolist()
        predictions = [(test_tweets0[j][0], int(predictions[j]), test_tweets0[j][2]) for j in range(len(test_tweets0))]
        save_results_to_csv(predictions, "../results/cnn-feats-svm.csv")
