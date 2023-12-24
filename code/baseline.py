"""
Classifies a tweet based on the number of positive and negative words in it
"""

import utils
from var import *


def classify(processed_csv, test_file=True, **params):
    positive_words = utils.file_to_word_set(params.pop("positive_words"))
    negative_words = utils.file_to_word_set(params.pop("negative_words"))
    predictions_list = []
    with open(processed_csv, "r") as csv:
        for line in csv:
            if test_file:
                tweet_id, tweet = line.strip().split(",")
            else:
                tweet_id, label, tweet = line.strip().split(",")
            pos_count, neg_count = 0, 0
            for word in tweet.split():
                if word in positive_words:
                    pos_count += 1
                elif word in negative_words:
                    neg_count += 1
            prediction = 1 if pos_count >= neg_count else 0
            if test_file:
                predictions_list.append((tweet_id, prediction, tweet))
            else:
                predictions_list.append((tweet_id, int(label), prediction))
    return predictions_list


if __name__ == "__main__":
    train_prediction = input("Enter your choice: \n 1) Train \n 2) Prediction\n")
    if train_prediction == "1":
        predictions = classify(train_processed_file, test_file=False, positive_words=positive_words_file,
                               negative_words=negative_words_file)
        correct = sum([1 for p in predictions if p[1] == p[2]]) * 100.0 / len(predictions)
        print("Correct = %.2f%%" % correct)
    elif train_prediction == "2":
        predictions = classify(test_processed_file, test_file=True, positive_words=positive_words_file,
                               negative_words=negative_words_file)
        utils.save_results_to_csv(predictions, "../results/baseline.csv")
        print("Predictions saved to ../results/baseline.csv")
    else:
        print("Wrong Choice.")
        exit()
