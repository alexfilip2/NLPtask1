import subprocess
from Ngrams import *
from pathlib import Path
import operator
from itertools import product
import sys 

root_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'SVMlight')
svm_light_learn, svm_light_classify = os.path.join(root_dir, 'svm_learn'), os.path.join(root_dir, 'svm_classify')
data_root_dir = os.path.join(root_dir, 'dataset')
emb_data_file = os.path.join(data_root_dir, 'embeddings')
subprocess_stdout = open(os.path.join(data_root_dir,'Results.txt'), 'w')

def embedding_unigram(review_path, sentiment_class, cutoff_unigrams, ngram_selection):
    # sent_class is either '+1' or '-1'
    embedding = {}
    with open(review_path, "r", encoding='UTF-8') as review_file:
        for line in review_file:
            for word in line.split():
                if ngram_selection == 'presence' and word in embedding.keys(): continue

                if word in cutoff_unigrams.keys():
                    if word in embedding.keys():
                        embedding[word] += 1
                    else:
                        embedding[word] = 1

    str_embedd = '+1' if sentiment_class == 'positive' else '-1'
    sorted_features = sorted(cutoff_unigrams.items(), key=operator.itemgetter(1))
    for (unigram, id) in sorted_features:
        if unigram in embedding.keys():
            str_embedd += " " + str(id) + ":" + str(embedding[unigram])

    return str_embedd


def embedding_bigram(review_path, sentiment_class, cutoff_bigrams, ngram_selection):
    # sent_class is either '+1' or '-1'
    embedding = {}
    with open(review_path, "r", encoding='UTF-8') as review_file:
        first_word = review_file.readline().split()[0]
        for line in review_file:
            for word in line.split():
                second_word = word
                bigram = first_word + " " + second_word
                if ngram_selection == 'presence' and bigram in embedding.keys(): continue

                if bigram in cutoff_bigrams.keys():
                    if bigram in embedding.keys():
                        embedding[bigram] += 1
                    else:
                        embedding[bigram] = 1
                first_word = second_word

    str_embedd = '+1' if sentiment_class == 'positive' else '-1'
    sorted_features = sorted(cutoff_bigrams.items(), key=operator.itemgetter(1))
    for (bigram, id) in sorted_features:
        if bigram in embedding.keys():
            str_embedd += " " + str(id) + ":" + str(embedding[bigram])

    return str_embedd


def create_embedding_dataset(ngram_selection, ngram_type, u_cutoff, b_cutoff ):
    doc_dataset = split_RR_NB(-1, 1, len(os.listdir(pos_stem_dir)))['train']
    embedding_dataset = open(emb_data_file + '_' + ngram_selection + '_' + ngram_type, "w", encoding='UTF-8')

    if ngram_type == 'unigram':
        individual_features = get_cutoff_unigrams(at_least_times=u_cutoff, id_feature_start=1)
    elif ngram_type == 'bigram':
        individual_features = get_cutoff_bigrams(at_least_times=b_cutoff, id_feature_start=1)
    else:
        unigram_features = get_cutoff_unigrams(at_least_times=u_cutoff, id_feature_start=1)
        bigram_id_start = len(unigram_features.keys()) + 1
        bigram_features = get_cutoff_bigrams(at_least_times=b_cutoff, id_feature_start=bigram_id_start)
        for (path, r_class) in doc_dataset:
            embedding_str = embedding_unigram(path, r_class, unigram_features, ngram_selection) + " "
            # discrad the class label of the bigram embedding
            embedding_str += embedding_bigram(path, r_class, bigram_features, ngram_selection).split(' ', 1)[1]
            embedding_dataset.write(embedding_str + "\n")
        return

    embedding_function = embedding_unigram if ngram_type == 'unigram' else embedding_bigram
    for (review_path, rev_class) in doc_dataset:
        embedding_str = embedding_function(review_path, rev_class, individual_features, ngram_selection)
        embedding_dataset.write(embedding_str + "\n")


def split_RR_embeddings(test_fold_id, train_test_ratio, ngram_selection, ngram_type):
    model_type = '_' + ngram_selection + "_" + ngram_type
    train = open(os.path.join(data_root_dir, 'train' + model_type), "w", encoding='UTF-8')
    test = open(os.path.join(data_root_dir, 'test' + model_type), "w", encoding='UTF-8')
    with open(emb_data_file + model_type, "r", encoding='UTF-8') as dataset:
        for line_nr, line in enumerate(dataset, 0):
            if (line_nr % train_test_ratio) == test_fold_id:
                test.write(line)
            else:
                train.write(line)


def train(ngram_selection, ngram_type):
    # create model file
    model_type = '_' + ngram_selection + "_" + ngram_type
    train_file = os.path.join( data_root_dir, 'train' + model_type)
    model_path = os.path.join(data_root_dir, 'model' + model_type)
    # create the model file if not existing
    model_file = open(model_path, "w", encoding='UTF-8')
    # call the executable for SVM training
    subprocess.call((svm_light_learn + " -z c -m 100 " + train_file + " " + model_path), shell=True)
    model_file.close()


def evaluate(ngram_selection, ngram_type, split_nr):
    model_type = '_' + ngram_selection + "_" + ngram_type
    model_path = os.path.join(data_root_dir , 'model' + model_type)
    test_file = os.path.join(data_root_dir, 'test' + model_type)
    result_path = os.path.join(data_root_dir, 'prediction' + model_type + str(split_nr))
    # create the prediction result file if not existing
    results_file = open(result_path, "w", encoding='UTF-8')
    # call the executable for SVM testing
    subprocess.call((svm_light_classify + " " + test_file + " " + model_path + " " + result_path), shell=True, stdout = subprocess_stdout)
    results_file.close()


def cross_validation_SVM(nr_of_folds, ngram_selection, ngram_type):
    for iter in range(nr_of_folds):
        split_RR_embeddings(iter, nr_of_folds, ngram_selection, ngram_type)
        train(ngram_selection, ngram_type)
        evaluate(ngram_selection, ngram_type, iter)


def validation():
    for u_cutoff, b_cutoff in product(range(1, 4), range(2,7)):
        create_embedding_dataset('presence', 'unigram+bigram', u_cutoff, b_cutoff)
        print('The cutoffs are for unigrams ' + str(u_cutoff) + ' and for bigrams ' + str(b_cutoff))
        print('Results for the SVM model presence unigram+bigram')
        cross_validation_SVM(10, 'presence', 'unigram+bigram')
    return


def summary_results(nr_of_folds):
    result_file = open(os.path.join(data_root_dir,'Results.txt'), 'r', encoding='UTF-8')
    acc = 0
    fold = 0
    for line in result_file:

        if line.split()[0] == 'Accuracy':
            fold += 1
            acc += float(line.split()[4].split('%')[0])

        if fold == nr_of_folds:
            print('accuracy is ' + str(acc / nr_of_folds))
            acc = 0
            fold = 0


if __name__ == "__main__":
    models = [('presence', 'unigram'), ('frequency', 'unigram'), ('presence', 'bigram'), ('presence', 'unigram+bigram')]
    for ngram_selection, ngram_type in models:
        dataset = Path(emb_data_file + '_' + ngram_selection + '_' + ngram_type)
        if not dataset.is_file():
            create_embedding_dataset(ngram_selection, ngram_type, 4, 7)
        else:
            print('Already created ' + ngram_selection + ' ' + ngram_type + ' embeddings')
    
    print('Results for the SVM model frequency unigram')
    cross_validation_SVM(10, 'frequency', 'unigram')
    print('Results for the SVM model presence unigram')
    cross_validation_SVM(10, 'presence', 'unigram')    
    print('Results for the SVM model presence bigram')
    cross_validation_SVM(10, 'presence', 'bigram')
    print('Results for the SVM model presence unigram+bigram')
    cross_validation_SVM(10, 'presence', 'unigram+bigram')
   
    #validation()
    summary_results(10)
    
