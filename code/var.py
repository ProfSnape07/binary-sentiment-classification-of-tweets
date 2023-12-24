freq_dist_file = "../dataset/twitter_training-processed-uni-freq-dist.pkl"
bi_freq_dist_file = "../dataset/twitter_training-processed-bi-freq-dist.pkl"
train_processed_file = "../dataset/twitter_training-processed.csv"
test_processed_file = "../dataset/twitter_validation-processed.csv"
positive_words_file = "../dataset/negative-words.txt"
negative_words_file = "../dataset/positive-words.txt"
glove_file_path = "../dataset/glove.twitter.27B.200d.txt"


uni_gram_size = 15000
vocab_size = uni_gram_size

# If using bigrams.
use_bi_grams = True
if use_bi_grams:
    bi_gram_size = 10000
    vocab_size += bi_gram_size

feat_type = "frequency"
