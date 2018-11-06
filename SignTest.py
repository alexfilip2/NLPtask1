import math
from Ngrams import *


def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def extract_result(model, test_fold_nr):
    results = []
    if model['type'] == 'NB':
        result_path = '../NLPtask1/NaiveBayes/prediction' + "_" + model['ngram_selection'] + "_" + model[
            'ngram_type'] + str(test_fold_nr)
        with open(result_path, 'r', encoding='UTF-8') as result_file:
            for line in result_file:
                for word in line.split():
                    results.append(word)
    else:
        result_path = '../NLPtask1/SVMlight/dataset/prediction' + "_" + model['ngram_selection'] + "_" + model[
            'ngram_type'] + str(test_fold_nr)
        with open(result_path, 'r', encoding='UTF-8') as result_file:
            for line in result_file:
                for word in line.split():
                    results.append('positive' if float(word) > 0 else 'negative')

    return results


def compute_p_value(model1, model2, test_fold_nr):
    null, minus, plus, p_value = 0, 0, 0, 0
    m1_results, m2_results = extract_result(model1, test_fold_nr), extract_result(model2, test_fold_nr)

    testset = split_RR_dataset(test_fold_nr, 10, 1000, True)

    for (review, true_sent), result1, result2 in zip(testset['test'], m1_results, m2_results):
        if result1 == result2:
            null += 1
        elif result1 == true_sent:
            plus += 1
        elif result2 == true_sent:
            minus += 1

    n = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)

    print(n, k, null, minus, plus)

    for iterator in range(0, k + 1):
        p_value += ncr(n, iterator)
    p_value = (p_value * 100) / pow(2, n - 1)

    return p_value


if __name__ == "__main__":
    p_val = 0
    for fold_nr in range(10):

        print(compute_p_value({'type': 'NB', 'ngram_selection': 'frequency', 'ngram_type': 'unigram'},
                              {'type': 'NB', 'ngram_selection': 'frequency', 'ngram_type': 'unigram'}, fold_nr))

    print(p_val / 10)
