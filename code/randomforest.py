"""
Performs classification using RandomForest classifier.
"""
from sklearn.ensemble import RandomForestClassifier
from utils import *

feat_type = "presence"


if __name__ == "__main__":
    train_prediction = input("Enter your choice: \n 1) Train \n 2) Prediction\n")
    if train_prediction == "1":
        train = True
    elif train_prediction == "2":
        train = False
    else:
        print("Wrong option.")
        exit()

    if train:
        tweets = process_tweets(train_processed_file, test_file=False)
        train_tweets, val_tweets = split_data(tweets)

        # Training
        classifier = RandomForestClassifier(n_jobs=2, random_state=0)
        train_model(train_tweets, classifier)

        # Validating
        validate_model(val_tweets, classifier)

        # Saving
        with open("../models/random-forest.pkl", "wb") as file:
            pickle.dump(classifier, file)
            file.close()
        print(f"\nSaved to ../models/random-forest.pkl")

    else:
        classifier = open_model("../models/random-forest.pkl")

        test_tweets = process_tweets(test_processed_file, test_file=True)
        predictions = predict_using_model(test_tweets, classifier)

        save_results_to_csv(predictions, "../results/random-forest.csv")
