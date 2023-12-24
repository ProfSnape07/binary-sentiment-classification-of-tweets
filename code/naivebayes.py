"""
Performs classification using Naive Bayes.
"""

from sklearn.naive_bayes import MultinomialNB
from utils import is_train, process_tweets, split_data, extract_features, apply_tf_idf, validate_model, open_model, \
    predict_using_model, save_results_to_csv
from var import *
import pickle

if __name__ == "__main__":
    train = is_train()

    if train:
        tweets = process_tweets(train_processed_file, test_file=False)
        train_tweets, val_tweets = split_data(tweets)

        # Training
        classifier = MultinomialNB()
        print("Extracting features & training model")
        batch_size = len(train_tweets)
        for training_set_X, training_set_y in extract_features(train_tweets, test_file=False, feat_type=feat_type,
                                                               batch_size=batch_size):
            if feat_type == "frequency":
                tfidf = apply_tf_idf(training_set_X)
                training_set_X = tfidf.transform(training_set_X)
            classifier.partial_fit(training_set_X, training_set_y, classes=[0, 1])
        print("\nTraining complete")

        # Validating
        validate_model(val_tweets, classifier)

        # Saving
        with open("../models/naivebayes.pkl", "wb") as file:
            pickle.dump(classifier, file)
            file.close()

    else:
        classifier = open_model("../models/naivebayes.pkl")

        test_tweets = process_tweets(test_processed_file, test_file=True)
        predictions = predict_using_model(test_tweets, classifier)

        save_results_to_csv(predictions, "../results/naivebayes.csv")
