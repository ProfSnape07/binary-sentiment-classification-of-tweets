import os
import glob
import pandas as pd


def majority_vote():
    csv_files = glob.glob("../results/*.csv")
    skip_files = {"../results/majority-voting.csv", "../results/baseline.csv"}
    skip_files = set(map(os.path.normpath, skip_files))

    combined_df = pd.DataFrame()
    for file in csv_files:
        if os.path.normpath(file) in skip_files:
            continue
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df])

    prediction_counts = {}
    for index, row in combined_df.iterrows():
        tweet_id = row["tweet_id"]
        prediction = row["prediction"]
        tweet = row["tweet"]

        if tweet_id not in prediction_counts:
            prediction_counts[tweet_id] = {"0": 0, "1": 0, "tweet": tweet}

        prediction_counts[tweet_id][str(int(prediction))] += 1

    majority_predictions = {}
    for tweet_id, prediction_count in prediction_counts.items():
        if prediction_count["0"] > prediction_count["1"]:
            majority_prediction = 0
        else:
            majority_prediction = 1

        majority_predictions[tweet_id] = majority_prediction

    result_df = []
    for key, value in majority_predictions.items():
        row = [key, value, prediction_counts[key]["tweet"]]
        result_df.append(row)

    result_df = pd.DataFrame(result_df, columns=["tweet_id", "prediction", "tweet"])

    result_df.to_csv("../results/majority-voting.csv", index=False)
    print("Predictions saved to ../results/majority-voting.csv")


if __name__ == "__main__":
    majority_vote()
