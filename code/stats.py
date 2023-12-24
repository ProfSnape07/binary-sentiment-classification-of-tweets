"""
Takes in a preprocessed CSV file and gives statistics
Writes the frequency distribution of words and bigrams
to pickle files.
"""

from nltk import FreqDist
import pickle
from utils import write_status
from collections import Counter


def analyze_tweet(given_tweet):
    result_ = {"MENTIONS": given_tweet.count("USER_MENTION"), "URLS": given_tweet.count("URL"),
               "POS_EMOS": given_tweet.count("EMO_POS"), "NEG_EMOS": given_tweet.count("EMO_NEG")}
    given_tweet = given_tweet.replace("USER_MENTION", "").replace("URL", "")
    words_ = given_tweet.split()
    result_["WORDS"] = len(words_)
    bi_grams = get_bigrams(words_)
    result_["BIGRAMS"] = len(bi_grams)
    return result_, words_, bi_grams


def get_bigrams(tweet_words):
    bi_grams = []
    num_of_words = len(tweet_words)
    for _ in range(num_of_words - 1):
        bi_grams.append((tweet_words[_], tweet_words[_ + 1]))
    return bi_grams


def get_bi_gram_freq_dist(bi_grams):
    freq_dict = {}
    for bi_gram in bi_grams:
        if freq_dict.get(bi_gram):
            freq_dict[bi_gram] += 1
        else:
            freq_dict[bi_gram] = 1
    counter = Counter(freq_dict)
    return counter


if __name__ == "__main__":
    csv_file_name = "../dataset/twitter_training-processed.csv"
    print(f"Working on {csv_file_name}")
    num_pos_tweets, num_neg_tweets = 0, 0
    num_mentions, max_mentions = 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls = 0, 0
    num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
    num_bi_grams, num_unique_bigrams = 0, 0
    all_words = []
    all_bigrams = []

    with open(csv_file_name, "r") as csv:
        lines = csv.readlines()
        num_tweets = len(lines)
        for i, line in enumerate(lines):
            t_id, if_pos, tweet = line.strip().split(",")
            if_pos = int(if_pos)
            if if_pos:
                num_pos_tweets += 1
            else:
                num_neg_tweets += 1
            result, words, bigrams = analyze_tweet(tweet)
            num_mentions += result["MENTIONS"]
            max_mentions = max(max_mentions, result["MENTIONS"])
            num_pos_emojis += result["POS_EMOS"]
            num_neg_emojis += result["NEG_EMOS"]
            max_emojis = max(max_emojis, result["POS_EMOS"] + result["NEG_EMOS"])
            num_urls += result["URLS"]
            max_urls = max(max_urls, result["URLS"])
            num_words += result["WORDS"]
            min_words = min(min_words, result["WORDS"])
            max_words = max(max_words, result["WORDS"])
            all_words.extend(words)
            num_bi_grams += result["BIGRAMS"]
            all_bigrams.extend(bigrams)
            write_status(i + 1, num_tweets)
    num_emojis = num_pos_emojis + num_neg_emojis
    unique_words = list(set(all_words))
    num_unique_words = len(unique_words)
    unique_bigrams = list(set(all_bigrams))
    num_unique_bigrams = len(unique_bigrams)

    with open(csv_file_name[:-4] + "-uni-gram.txt", "w") as uwf:
        uwf.write('\n'.join(unique_words))
        uwf.close()

    with open(csv_file_name[:-4] + "-bi-gram.txt", "w") as uwf:
        unique_bi_gram = []
        for i in unique_bigrams:
            bi_gram1 = i[0] + " " + i[1]
            unique_bi_gram.append(bi_gram1)
        uwf.write('\n'.join(unique_bi_gram))
        uwf.close()

    print("\nCalculating frequency distribution")

    # Uni grams
    freq_dist = FreqDist(all_words)
    pkl_file_name = csv_file_name[:-4] + "-uni-freq-dist.pkl"
    with open(pkl_file_name, "wb") as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    print("Saved uni-frequency distribution to %s" % pkl_file_name)

    # Bigrams
    bi_gram_freq_dist = get_bi_gram_freq_dist(all_bigrams)
    bi_pkl_file_name = csv_file_name[:-4] + '-bi-freq-dist.pkl'
    with open(bi_pkl_file_name, 'wb') as pkl_file:
        pickle.dump(bi_gram_freq_dist, pkl_file)
    print("Saved bi-frequency distribution to %s" % bi_pkl_file_name)

    print("\n[Analysis Statistics]")
    print("Tweets => Total: %d, Positive: %d, Negative: %d" % (num_tweets, num_pos_tweets, num_neg_tweets))
    print("User Mentions => Total: %d, Avg: %.4f, Max: %d" % (
        num_mentions, num_mentions / float(num_tweets), max_mentions))
    print("URLs => Total: %d, Avg: %.4f, Max: %d" % (num_urls, num_urls / float(num_tweets), max_urls))
    print("Emojis => Total: %d, Positive: %d, Negative: %d, Avg: %.4f, Max: %d" % (
        num_emojis, num_pos_emojis, num_neg_emojis, num_emojis / float(num_tweets), max_emojis))
    print("Words => Total: %d, Unique: %d, Avg: %.4f, Max: %d, Min: %d" % (
        num_words, num_unique_words, num_words / float(num_tweets), max_words, min_words))
    print("Bigrams => Total: %d, Unique: %d, Avg: %.4f" % (
        num_bi_grams, num_unique_bigrams, num_bi_grams / float(num_tweets)))
