import math
import numpy as np
from Ngrams import *
import sys

nb_result_file = open(os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'NaiveBayesAccuracies'), 'w', encoding='UTF-8')
std_out = sys.stdout
sys.stdout = nb_result_file

nb_dataset = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'NaiveBayes')
if not os.path.exists(nb_dataset):
    os.makedirs(nb_dataset)

selected_unigrams = get_cutoff_unigrams(4, 1).keys()
selected_bigrams = get_cutoff_bigrams(7, 1).keys()


# training set is a list of pairs: (string path of review, class)
# smooth = smoothing constant
# ngram_type ='unigram'/'bigram'/'unigram+bigram'
def smoothed_ngram_probs(trainingset, smooth, ngram_type):
    pos_probs, neg_probs = {}, {}
    ngram_counter, selected_ngrams = (unigram_class_count, selected_unigrams) if ngram_type == 'unigram' else (
        bigram_class_count, selected_bigrams)

    counts = ngram_counter(trainingset)
    pos_count, neg_count = counts['pcount'], counts['ncount']

    total_count_pos = np.sum([pos_count[ngram] + smooth for ngram in pos_count.keys() if ngram in selected_ngrams])
    total_count_neg = np.sum([neg_count[ngram] + smooth for ngram in neg_count.keys() if ngram in selected_ngrams])

    for ngram in pos_count.keys():
        if ngram not in selected_ngrams: continue
        pos_probs[ngram] = (pos_count[ngram] + smooth) / total_count_pos
        if ngram not in neg_count.keys():
            neg_probs[ngram] = smooth / total_count_neg

    for ngram in neg_count.keys():
        if ngram not in selected_ngrams: continue
        neg_probs[ngram] = (neg_count[ngram] + smooth) / total_count_neg
        if ngram not in pos_count.keys():
            pos_probs[ngram] = smooth / total_count_pos

    return {'p_prob': pos_probs, 'n_prob': neg_probs}


# test_file_path = string path of a review for testing
# unigram_probs = probability of unigram feature given review class
# unigram_selection = 'frequency'/'presence' of feature
def predict_with_unigrams(test_rev_path, unigram_probs, unigram_selection):
    rev_pos_prob, rev_neg_prob = 0, 0
    seen_words = set([])

    with open(test_rev_path, "r", encoding='UTF-8') as file:
        for line in file:
            for word in line.split():

                if unigram_selection == 'presence':
                    if word not in seen_words:
                        seen_words.add(word)
                    else:
                        continue

                if word in unigram_probs['p_prob'].keys() and unigram_probs['p_prob'][word] > 0:
                    rev_pos_prob += math.log10(unigram_probs['p_prob'][word])
                if word in unigram_probs['n_prob'].keys() and unigram_probs['n_prob'][word] > 0:
                    rev_neg_prob += math.log10(unigram_probs['n_prob'][word])

    return {'pos': rev_pos_prob, 'neg': rev_neg_prob}


def predict_with_bigrams(test_file_path, bigram_probs, bigram_selection):
    rev_pos_prob, rev_neg_prob = 0, 0
    seen_bigrams = set([])
    with open(test_file_path, "r", encoding='UTF-8') as review_file:
        first_word = review_file.readline()
        for line in review_file:
            for word in line.split():
                second_word = word
                bigram = first_word + " " + second_word

                if bigram_selection == 'presence':
                    if bigram not in seen_bigrams:
                        seen_bigrams.add(bigram)
                    else:
                        first_word = second_word
                        continue

                if bigram in bigram_probs['p_prob'].keys() and bigram_probs['p_prob'][bigram] > 0:
                    rev_pos_prob += math.log10(bigram_probs['p_prob'][bigram])
                if bigram in bigram_probs['n_prob'].keys() and bigram_probs['n_prob'][bigram] > 0:
                    rev_neg_prob += math.log10(bigram_probs['n_prob'][bigram])
                first_word = second_word

    return {'pos': rev_pos_prob, 'neg': rev_neg_prob}


# prediction = predict_with_unigrams/predict_with_bigrams
def accuracy(trainset, testset, ngram_selection, ngram_type, smooth):
    acc = 0
    predict_results = []
    if ngram_type == 'unigram+bigram':
        unigram_probs = smoothed_ngram_probs(trainset, smooth=smooth, ngram_type='unigram')
        bigram_probs = smoothed_ngram_probs(trainset, smooth=smooth, ngram_type='bigram')

        for (test_file, sentiment) in testset:
            decision_unigram = predict_with_unigrams(test_file, unigram_probs, ngram_selection)
            decision_bigram = predict_with_bigrams(test_file, bigram_probs, ngram_selection)
            if (decision_unigram['pos'] + decision_bigram['pos']) < (decision_unigram['neg'] + decision_bigram['neg']):
                result = 'negative'
            else:
                result = 'positive'
            if result == sentiment: acc += 1

            predict_results.append(result)
    else:
        ngram_probs = smoothed_ngram_probs(trainingset=trainset, smooth=smooth, ngram_type=ngram_type)
        predict = predict_with_unigrams if ngram_type == 'unigram' else predict_with_bigrams
        for (test_file, sentiment) in testset:
            decision_prob = predict(test_file, ngram_probs, ngram_selection)
            if decision_prob['neg'] > decision_prob['pos']:
                result = 'negative'
            else:
                result = 'positive'

            if result == sentiment: acc += 1
            predict_results.append(result)

    return {'acc_value': acc / len(testset), 'results': predict_results}


def tenfold_RR_cv(nr_of_folds, ngram_selection, ngram_type, smooth):
    accuracy_avg = 0
    limit = len(os.listdir(pos_stem_dir))

    if ngram_type == 'unigram':
        feat_number = len(selected_unigrams)
    elif ngram_type == 'bigram':
        feat_number = len(selected_bigrams)
    else:
        feat_number = len(selected_unigrams) + len(selected_bigrams)
    print("The number of features used is: " + str(feat_number))

    for iter in range(nr_of_folds):
        result_path = os.path.join(nb_dataset,
                                   'prediction' + "_" + ngram_selection + "_" + ngram_type + str(smooth) + str(iter))
        result_file = open(result_path, 'w', encoding='UTF-8')

        dataset = split_RR_NB(iter, nr_of_folds, limit)
        accuracy_result = accuracy(dataset['train'], dataset['test'], ngram_selection, ngram_type, smooth)
        current_iter_accuracy, predictions = accuracy_result['acc_value'], accuracy_result['results']

        for result in predictions: result_file.write(result + "\n")
        result_file.close()

        accuracy_avg += current_iter_accuracy

    return accuracy_avg / nr_of_folds


if __name__ == "__main__":
    models = [('presence', 'unigram'), ('frequency', 'unigram'), ('presence', 'bigram'), ('presence', 'unigram+bigram')]
    for ngram_selection, ngram_type in models:
        print('The accuracy using cross-validation with sNB using: ' + ngram_type + " based on " + ngram_selection)
        print(tenfold_RR_cv(10, ngram_selection, ngram_type, 1))
        print()
        print('The accuracy using cross-validation with NB using: ' + ngram_type + " based on " + ngram_selection)
        print(tenfold_RR_cv(10, ngram_selection, ngram_type, 0))
        print()
    print('Evaluation is complete. The results have been written to ' + 'NLPtask1/NaiveBayesAccuracies', file=std_out)
    nb_result_file.close()
