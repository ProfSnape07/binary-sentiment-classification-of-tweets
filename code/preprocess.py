import re
from utils import write_status
from nltk.stem.porter import PorterStemmer


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return re.search(r'^[a-zA-Z][a-z0-9A-Z._]*$', word) is not None


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.\S+)|(https?://\S+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@\S+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def preprocess_csv(old_csv_file_name, new_csv_file_name, test_file=False):
    save_to_file = open(new_csv_file_name, "w")
    with open(old_csv_file_name, "r", encoding="utf-8") as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(",")]
            if not test_file:
                line = line[1 + line.find(","):]
                positive = int(line[:line.find(",")])
            line = line[1 + line.find(","):]
            tweet = line
            processed_tweet = preprocess_tweet(tweet)
            if not test_file:
                save_to_file.write("%s,%d,%s\n" % (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write("%s,%s\n" % (tweet_id, processed_tweet))
            write_status(i + 1, total)
    save_to_file.close()
    print("\nSaved processed tweets to: %s" % new_csv_file_name)
    return


if __name__ == "__main__":
    using_stemmer = input("Do you want to use stemmer: \n 1) Yes \n 2) No\n")
    if using_stemmer == "1":
        use_stemmer = True
    elif using_stemmer == "2":
        use_stemmer = False
    else:
        print("Wrong Choice.")
        exit()

    file_name = input("Enter file choice (1/2): \n 1) twitter_training.csv \n 2) twitter_validation.csv\n")

    if file_name == "1":
        csv_file_name = "../dataset/twitter_training.csv"
        processed_file_name = csv_file_name[:-4] + "-processed.csv"
        if use_stemmer:
            porter_stemmer = PorterStemmer()
            processed_file_name = csv_file_name[:-4] + "-processed-stemmed.csv"

        preprocess_csv(csv_file_name, processed_file_name, test_file=False)

    elif file_name == "2":
        csv_file_name = "../dataset/twitter_validation.csv"
        processed_file_name = csv_file_name[:-4] + "-processed.csv"
        if use_stemmer:
            porter_stemmer = PorterStemmer()
            processed_file_name = csv_file_name[:-4] + "-processed-stemmed.csv"

        preprocess_csv(csv_file_name, processed_file_name, test_file=True)

    else:
        print("Wrong Choice.")
        exit()
