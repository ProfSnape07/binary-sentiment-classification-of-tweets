"""
Performs classification using SVM.
"""
from sklearn import svm
import pickle
from var import *
from utils import process_tweets, split_data, train_model, validate_model, open_model, predict_using_model, is_train, \
    save_results_to_csv

if __name__ == "__main__":
    feat_type = "frequency"

    train = is_train()

    if train:
        tweets = process_tweets(train_processed_file, test_file=False)
        train_tweets, val_tweets = split_data(tweets)

        # Training
        classifier = svm.LinearSVC(C=0.1)
        train_model(train_tweets, classifier)

        # Validating
        validate_model(val_tweets, classifier)

        # Saving
        with open("../models/svm.pkl", "wb") as file:
            pickle.dump(classifier, file)
            file.close()
        print(f"\nSaved to ../models/svm.pkl")

    else:
        classifier = open_model("../models/svm.pkl")

        test_tweets = process_tweets(test_processed_file, test_file=True)
        predictions = predict_using_model(test_tweets, classifier)

        save_results_to_csv(predictions, "../results/svm.csv")
