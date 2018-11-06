import math
import numpy as np
from Ngrams import *
from itertools import product


# training set is a list of pairs: (string path of review, class)
# smooth = smoothing constant
# ngram_type ='unigram'/'bigram'/'unigram+bigram'
def smoothed_ngram_probs(trainingset, smooth, ngram_type):
    pos_probs, neg_probs = {}, {}
    ngram_counter, selected_ngrams = (unigram_class_count, get_unigrams(4, 1).keys()) if ngram_type == 'unigram' else (
        bigram_class_count, (get_bigrams(7, 1).keys()))

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

    return {'pprob': pos_probs, 'nprob': neg_probs}


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

                if word in unigram_probs['pprob'].keys() and unigram_probs['pprob'][word] > 0:
                    rev_pos_prob += math.log10(unigram_probs['pprob'][word])
                if word in unigram_probs['nprob'].keys() and unigram_probs['nprob'][word] > 0:
                    rev_neg_prob += math.log10(unigram_probs['nprob'][word])

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

                if bigram in bigram_probs['pprob'].keys() and bigram_probs['pprob'][bigram] > 0:
                    rev_pos_prob += math.log10(bigram_probs['pprob'][bigram])
                if bigram in bigram_probs['nprob'].keys() and bigram_probs['nprob'][bigram] > 0:
                    rev_neg_prob += math.log10(bigram_probs['nprob'][bigram])
                first_word = second_word

    return {'pos': rev_pos_prob, 'neg': rev_neg_prob}


# prediction = predict_with_unigrams/predict_with_bigrams
def accuracy(trainset, testset, ngram_selection, ngram_type, prediction):
    acc = 0
    predictions = []
    if ngram_type == 'unigram+bigram':
        unigram_probs = smoothed_ngram_probs(trainset, smooth=1, ngram_type='unigram')
        bigram_probs = smoothed_ngram_probs(trainset, smooth=1, ngram_type='bigram')

        for (test_file, sentiment) in testset:
            decision_unigram = predict_with_unigrams(test_file, unigram_probs, ngram_selection)
            decision_bigram = predict_with_bigrams(test_file, bigram_probs, ngram_selection)
            if (decision_unigram['pos'] + decision_bigram['pos']) < (decision_unigram['neg'] + decision_bigram['neg']):
                result = 'negative'
            else:
                result = 'positive'
            if result == sentiment: acc += 1
            predictions.append(result)
    else:
        ngram_probs = smoothed_ngram_probs(trainingset=trainset, smooth=1, ngram_type=ngram_type)
        for (test_file, sentiment) in testset:
            decision_prob = prediction(test_file, ngram_probs, ngram_selection)
            result = 'negative' if decision_prob['neg'] > decision_prob['pos'] else 'positive'
            if result == sentiment: acc += 1
            predictions.append(result)

    return {'acc': acc / len(testset), 'predictions': predictions}


def tenfold_RR_cv(metric, fold_nr, stem_flag, ngram_selection, ngram_type):
    metric_avg = 0

    prediction = predict_with_unigrams if ngram_type == 'unigram' else predict_with_bigrams
    class_rev_size = len(os.listdir(pos_rev_dir))

    for i in range(fold_nr):
        result_path = '../NLPtask1/NaiveBayes/prediction' + "_" + ngram_selection + "_" + ngram_type + str(i)
        result_file = open(result_path, 'w', encoding='UTF-8')
        dataset = split_RR_dataset(i, 10, class_rev_size, stem_flag)

        metric_res = metric(dataset['train'], dataset['test'], ngram_selection, ngram_type,
                            prediction)
        current_fold_score, predictions = metric_res['acc'], metric_res['predictions']
        for result in predictions:
            result_file.write(result + "\n")
        metric_avg += current_fold_score
    return metric_avg / fold_nr


if __name__ == "__main__":
    for flag, selection, ngram_type in product({True}, {'presence', 'frequency'},
                                               {'unigram', 'bigram', 'unigram+bigram'}):
        print('The accuracy using cross-validation with NB using: ' + ('stemmed ' if flag else 'not stemmed ')
              + 'dataset, ' + ngram_type + " based on " + selection)
        print(tenfold_RR_cv(accuracy, 10, flag, selection, ngram_type))
