import matplotlib.pyplot as plt
import numpy as np
import pickle

FREQ_DIST_FILE = "../dataset/twitter_training-processed-uni-freq-dist.pkl"
BI_FREQ_DIST_FILE = "../dataset/twitter_training-processed-bi-freq-dist.pkl"

with open(FREQ_DIST_FILE, 'rb') as pkl_file:
    freq_dist = pickle.load(pkl_file)
unigrams = freq_dist.most_common(20)
plt.figure(1, [10, 7])
x = np.array(range(0, 40, 2))
y = np.array([i[1] for i in unigrams])
my_xticks = [i[0] for i in unigrams]
plt.xticks(x, my_xticks, rotation=90)
plt.plot(x, y)
plt.xlabel("Words")
plt.ylabel("Occurrence")
plt.title("frequency distribution of top 20 words")
plt.show()

with open(BI_FREQ_DIST_FILE, "rb") as pkl_file:
    freq_dist = pickle.load(pkl_file)
bigrams = freq_dist.most_common(20)
plt.figure(1, [10, 7])
x = np.array(range(0, 40, 2))
y = np.array([i[1] for i in bigrams])
my_xticks = [", ".join(i[0]) for i in bigrams]
plt.xticks(x, my_xticks, rotation=90)
plt.plot(x, y)
plt.xlabel("Bigrams")
plt.ylabel("Occurrence")
plt.title("frequency distribution of top 20 bigrams")
plt.show()

with open(FREQ_DIST_FILE, "rb") as pkl_file:
    freq_dist = pickle.load(pkl_file)
unigrams = freq_dist.most_common(100)
log_ranks = np.log(range(1, 101))
log_freqs = np.log([i[1] for i in unigrams])
z = np.polyfit(log_ranks, log_freqs, 1)
p = np.poly1d(z)
plt.figure(3, [8, 6])
plt.plot(log_ranks, log_freqs, "ro")
plt.plot(log_ranks, p(log_ranks), "b-")
plt.xlabel("log (Rank)")
plt.ylabel("log (Frequency)")
plt.title("Zipf\'s Law")
plt.show()

classifiers = ['Baseline', 'Naive Bayes', 'MaxEnt', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM', 'MLP/Neuralnet',
               'LSTM', 'CNN', 'Logistic Regression', 'CNN feats SVM']
accuracies = [25.11, 82.29, 78.42, 71.50, 78.56, 78.28, 82.29, 80.35, 81.05, 82.43, 83.67, 95.02]
plt.figure(4, [10, 8])
plt.bar(range(len(classifiers)), accuracies, align='center', alpha=0.5)
plt.xticks(range(len(classifiers)), classifiers, rotation=90)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Comparison of Various Classifiers")
plt.show()
