"""
Performs classification using Maximum Entropy (MaxEnt) classifier, also known as a Logistic Regression classifier.
"""
import nltk
import pickle
from utils import save_results_to_csv, split_data, is_train, open_model
from var import *

use_bi_grams = False


def get_data_from_train_file(file_name):
    data = []
    with open(file_name, "r") as csv:
        lines = csv.readlines()
        for _, line in enumerate(lines):
            sentiment = line.split(",")[1]
            bag_of_words = line.split(",")[2].split()
            if use_bi_grams:
                bag_of_words_bi_gram = list(nltk.bigrams(line.split(",")[2].split()))
                bag_of_words += bag_of_words_bi_gram
            data.append((bag_of_words, sentiment))
    return data


def process_tweets(csv_file):
    tweets = []
    with open(csv_file, "r") as csv:
        lines = csv.readlines()
        for _, line in enumerate(lines):
            tweet_id, tweet = line.split(",")
            bag_of_words = tweet.split()
            if use_bi_grams:
                bag_of_words_bi_gram = list(nltk.bigrams(tweet.split()))
                bag_of_words = bag_of_words + bag_of_words_bi_gram
            tweets.append((tweet_id, list_to_dict(bag_of_words), tweet))
    return tweets


def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])


if __name__ == "__main__":
    train = is_train()

    if train:
        train_data = get_data_from_train_file(train_processed_file)
        train_set, validation_set = split_data(train_data)
        print("Extracting features & training model")
        training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in train_set]
        validation_set_formatted = [(list_to_dict(element[0]), element[1]) for element in validation_set]
        numIterations = 1
        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[1]
        classifier = nltk.MaxentClassifier.train(training_set_formatted, algorithm, max_iter=numIterations)
        print("\nTraining complete")

        # Validating
        print("\nValidating model")
        count = 0
        for review in validation_set_formatted:
            label = review[1]
            text = review[0]
            determined_label = classifier.classify(text)
            if determined_label != label:
                count += 1
        accuracy = (len(validation_set) - count) * 100 / len(validation_set)
        print("Accuracy: %.4f %%" % accuracy)

        with open("../models/max-entropy.pkl", "wb") as file:
            pickle.dump(classifier, file)
            file.close()
        print(f"\nSaved to ../models/max-entropy.pkl")

    else:
        classifier = open_model("../models/max-entropy.pkl")

        test_data = process_tweets(test_processed_file)
        predictions = []
        for i in test_data:
            prediction = [i[0], classifier.classify(i[1]), i[2]]
            predictions.append(prediction)

        save_results_to_csv(predictions, "../results/max-entropy.csv")
