# importing the pandas library
import pandas as pd

file_name = input("Enter file choice: \n 1) twitter_training.csv \n 2) twitter_validation.csv\n")
if file_name == "1":
    csv_file_name = "../dataset/twitter_training.csv"
elif file_name == "2":
    csv_file_name = "../dataset/twitter_validation.csv"
else:
    print("Wrong option.")
    exit()

# reading the csv file
df = pd.read_csv(csv_file_name)

print(df)

# Assign column names to the DataFrame
df.columns = ["tweet_id", "entity", "sentiment", "tweet"]
df = df.drop('entity', axis=1)

value_mapping = {'Positive': 1, 'Negative': 0}
df["sentiment"] = df["sentiment"].replace(value_mapping)

# # Define a condition to drop rows based on a specific column
condition = df["sentiment"].isin([0, 1])

# Drop rows where "sentiment" is "Neutral" or "Irrelevant"
# Use boolean indexing to drop rows based on the condition
df = df[condition]

# Use drop_duplicates to keep only unique values in "tweet_id"
df = df.drop_duplicates(subset=["tweet_id"])

# Replace new lines with spaces in "tweet"
df["tweet"] = df["tweet"].str.replace('\n', ' ')

# writing into the file
df.to_csv(csv_file_name, header=False, index=False)

print(df)
