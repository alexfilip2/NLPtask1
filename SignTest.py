import math
from Ngrams import *

st_result_file = open(os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'SignTestResults'), 'w', encoding='UTF-8')
std_out = sys.stdout
sys.stdout = st_result_file


def comb(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


def extract_result(model, test_fold_nr):
    results = []
    if model['type'] == 'sNB' or model['type'] == 'NB':
        smooth = '0' if model['type'] == 'NB' else '1'
        naive_dataset = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'NaiveBayes')
        result_path = os.path.join(naive_dataset, 'prediction' + "_" + model['ngram_selection'] + "_" + model[
            'ngram_type'] + smooth + str(test_fold_nr))
        with open(result_path, 'r', encoding='UTF-8') as result_file:
            for line in result_file:
                for word in line.split():
                    results.append(word)
    else:
        svm_dataset = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'SVMlight', 'dataset')
        result_path = os.path.join(svm_dataset, 'prediction' + "_" + model['ngram_selection'] + "_" + model[
            'ngram_type'] + str(test_fold_nr))
        with open(result_path, 'r', encoding='UTF-8') as result_file:
            for line in result_file:
                for word in line.split():
                    results.append('positive' if float(word) > 0 else 'negative')

    return results


def compute_p_value(model1, model2, test_fold_nr):
    null, minus, plus, p_value = 0, 0, 0, 0
    m1_results, m2_results = extract_result(model1, test_fold_nr), extract_result(model2, test_fold_nr)

    testset = split_RR_NB(test_fold_nr, 10, 1000)

    for (review, true_sent), result1, result2 in zip(testset['test'], m1_results, m2_results):
        if result1 == result2:
            null += 1
        elif result1 == true_sent:
            plus += 1
        elif result2 == true_sent:
            minus += 1

    n = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    for iterator in range(k + 1):
        p_value += comb(n, iterator)
    p_value = (p_value) / pow(2, n - 1)

    return p_value


def model(str_descr):
    return {'type': str_descr.split()[0], 'ngram_selection': str_descr.split()[1], 'ngram_type': str_descr.split()[2]}


if __name__ == "__main__":
    interest_comp = [('sNB presence unigram', 'NB presence unigram'),
                     ('sNB frequency unigram', 'sNB presence unigram'),
                     ('SVM presence unigram', 'SVM frequency unigram'),
                     ('sNB presence unigram+bigram', 'SVM presence unigram+bigram')]

    for model1, model2 in interest_comp:
        print('The sign test results for ' + model1 + ' vs ' + model2)
        for split in range(10):
            print('p-value on split ' + str(split) + ' is:' + str(compute_p_value(model(model1), model(model2), split)))
        print()
    print('Sign test is complete. The results have been written to ' + 'NLPtask1/SignTestResults', file=std_out)
