import subprocess
from Ngrams import *
from pathlib import Path
import operator
import numpy as np
import matplotlib.pyplot as plt
root_dir = 'C:\\Users\\user\\PycharmProjects\\NLPtask1\\SVMlight\\'
svm_light_learn, svm_light_classify = root_dir + 'svm_learn.exe', root_dir + 'svm_classify.exe'
data_root_dir = root_dir + "dataset\\"
emb_data_file = data_root_dir + 'embeddings'


def embedding_unigram(review_path, sentiment_class, all_words, ngram_selection):
    # sent_class is either '+1' or '-1'
    embedding = {}
    with open(review_path, "r", encoding='UTF-8') as review_file:
        for line in review_file:
            for word in line.split():
                if ngram_selection == 'presence' and word in embedding.keys(): continue

                if word in all_words.keys():
                    if word in embedding.keys():
                        embedding[word] += 1
                    else:
                        embedding[word] = 1

    str_embedd = '+1' if sentiment_class == 'positive' else '-1'
    sorted_features = sorted(all_words.items(), key=operator.itemgetter(1))
    for (unigram, id) in sorted_features:
        if unigram in embedding.keys():
            str_embedd += " " + str(id) + ":" + str(embedding[unigram])

    return str_embedd


def embedding_bigram(review_path, sentiment_class, all_bigrams, ngram_selection):
    # sent_class is either '+1' or '-1'
    embedding = {}
    with open(review_path, "r", encoding='UTF-8') as review_file:
        first_word = review_file.readline().split()[0]
        for line in review_file:
            for word in line.split():
                second_word = word
                bigram = first_word + " " + second_word
                if ngram_selection == 'presence' and bigram in embedding.keys(): continue

                if bigram in all_bigrams.keys():
                    if bigram in embedding.keys():
                        embedding[bigram] += 1
                    else:
                        embedding[bigram] = 1
                first_word = second_word

    str_embedd = '+1' if sentiment_class == 'positive' else '-1'
    sorted_features = sorted(all_bigrams.items(), key=operator.itemgetter(1))
    for (bigram, id) in sorted_features:
        if bigram in embedding.keys():
            str_embedd += " " + str(id) + ":" + str(embedding[bigram])

    return str_embedd


def create_embedding_dataset(ngram_selection, ngram_type):
    doc_dataset = split_RR_dataset(-1, 1, len(os.listdir(pos_stem_dir)), stem_flag=True)['train']
    sorted_dataset = doc_dataset.sort(key=operator.itemgetter(1))
    embedding_dataset = open(emb_data_file + '_' + ngram_selection + '_' + ngram_type, "w",
                             encoding='UTF-8')

    if ngram_type == 'unigram':
        individual_features = get_unigrams(at_least_times=4, id_feature_start=1)
    else:
        if ngram_type == 'bigram':
            individual_features = get_bigrams(at_least_times=7, id_feature_start=1)
        else:
            unigram_features = get_unigrams(at_least_times=4, id_feature_start=1)
            bigram_id_start = len(unigram_features.keys()) + 1
            bigram__features = get_bigrams(at_least_times=4, id_feature_start=bigram_id_start)
            for (review_path, rev_class) in sorted_dataset:
                embedding_str = embedding_unigram(review_path, rev_class, unigram_features, ngram_selection) + " "
                # discrad the class label of the bigram embedding
                embedding_str += \
                embedding_bigram(review_path, rev_class, bigram__features, ngram_selection).split(' ', 1)[1]
                embedding_dataset.write(embedding_str + "\n")
            return

    embedding_function = embedding_unigram if ngram_type == 'unigram' else embedding_bigram
    for (review_path, rev_class) in sorted_dataset:
        embedding_str = embedding_function(review_path, rev_class, individual_features, ngram_selection)
        embedding_dataset.write(embedding_str + "\n")


dataset = Path(emb_data_file + '_presence_' + 'unigram')
if not dataset.is_file():
    create_embedding_dataset('presence', 'unigram')
else:
    print("Already created presence unigram embeddings")

dataset = Path(emb_data_file + '_frequency_' + 'unigram')
if not dataset.is_file():
    create_embedding_dataset('frequency', 'unigram')
else:
    print("Already created frequency stemmed embeddings")

dataset = Path(emb_data_file + '_presence_' + 'bigram')
if not dataset.is_file():
    create_embedding_dataset('presence', 'bigram')
else:
    print("Already created presence bigram embeddings")

dataset = Path(emb_data_file + '_presence_' + 'unigram+bigram')
if not dataset.is_file():
    create_embedding_dataset('presence', 'unigram+bigram')
else:
    print("Already created presence unigram+bigram embeddings")


def split_emb_data(test_fold_id, train_test_ratio, ngram_selection, ngram_type):
    line_nr = 0
    model_type = '_' + ngram_selection + "_" + ngram_type
    train = open(data_root_dir + 'train' + model_type, "w", encoding='UTF-8')
    test = open(data_root_dir + 'test' + model_type, "w", encoding='UTF-8')
    with open(emb_data_file + model_type, "r", encoding='UTF-8') as dataset:
        for line in dataset:
            if (line_nr % train_test_ratio) == test_fold_id:
                test.write(line)
            else:
                train.write(line)
            line_nr += 1


def train(ngram_selection, ngram_type):
    # create model file
    model_type = '_' + ngram_selection + "_" + ngram_type
    train_file = data_root_dir + 'train' + model_type
    model_path = data_root_dir + 'model' + model_type
    model_file = open(model_path, "w", encoding='UTF-8')

    subprocess.call((
            svm_light_learn + " -z c -m 100 " + train_file + " " + model_path), shell=True)
    model_file.close()


def evaluate(ngram_selection, ngram_type, split_nr):
    model_type = '_' + ngram_selection + "_" + ngram_type
    model_path = data_root_dir + 'model' + model_type
    test_file = data_root_dir + 'test' + model_type
    result_path = data_root_dir + 'prediction' + model_type + str(split_nr)
    results_file = open(result_path, "w", encoding='UTF-8')

    subprocess.call((svm_light_classify + " " + test_file + " " + model_path + " " + result_path), shell=True)
    results_file.close()


def cross_validation_SVM(fold_nr, ngram_selection, ngram_type):
    trained_model = Path(data_root_dir + 'model' + '_' + ngram_selection + "_" + ngram_type)
    for i in range(fold_nr):
        split_emb_data(i, fold_nr, ngram_selection, ngram_type)

        train(ngram_selection, ngram_type)
        evaluate(ngram_selection, ngram_type, i)

def summary_results():
    result_file = open('../NLPtask1/SVMlight/dataset/Results.txt', 'r', encoding='UTF-8')
    acc = 0
    fold = 0
    for line in result_file:
        model_type = ''
        if line.split()[0]=='Results':
            model_type = line.split()[-2] +' ' +line.split()[-1]
            print('For the SVM model ' +model_type)

        if line.split()[0] == 'Accuracy':
            fold+=1
            acc += float(line.split()[4].split('%')[0])

        if fold == 10:
            print('accuracy is '+ str(acc/10))
            acc =0
            fold = 0


if __name__ == "__main__":
    print('Results for the SVM model presence unigram')
    cross_validation_SVM(10, 'presence', 'unigram')
    print('Results for the SVM model frequency unigram')
    cross_validation_SVM(10, 'frequency', 'unigram')
    print('Results for the SVM model presence bigram')
    cross_validation_SVM(10, 'presence', 'bigram')
    print('Results for the SVM model presence unigram+bigram')
    cross_validation_SVM(10, 'presence', 'unigram+bigram')
    summary_results()
