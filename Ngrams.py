from stemming.porter2 import stem
import os
import operator
# pos_rev_dir = '../NLPtask1/POS'
pos_rev_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'POS')
# neg_rev_dir = '../NLPtask1/NEG'
neg_rev_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'NEG')
#pos_stem_dir = '../NLPtask1/StemPos'
pos_stem_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'StemPos')
#neg_stem_dir = '../NLPtask1/StemNeg'
neg_stem_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask1', 'StemNeg')

# stem the initial review dataset using the Porter stemming algorithm
def stem_all_reviews():
    if not os.path.exists(pos_stem_dir):
        os.makedirs(pos_stem_dir)
    else:
        assert len(sorted(os.listdir(pos_stem_dir))) == len(sorted(os.listdir(pos_rev_dir)))
        return
    if not os.path.exists(neg_stem_dir):
        os.makedirs(neg_stem_dir)
    else:
        assert len(sorted(os.listdir(neg_stem_dir))) == len(sorted(os.listdir(neg_rev_dir)))
        return

    pos_reviews = sorted(os.listdir(pos_rev_dir))
    neg_reviews = sorted(os.listdir(neg_rev_dir))

    for POS_review, NEG_review in zip(pos_reviews, neg_reviews):
        new_pos_stemmed = open(os.path.join(pos_stem_dir,'stemmed'+'_'+POS_review), "w", encoding='UTF-8')
        new_neg_stemmed = open(os.path.join(neg_stem_dir, 'stemmed' + '_' + NEG_review), "w", encoding='UTF-8')

        with open(os.path.join(pos_rev_dir, POS_review), "r", encoding='UTF-8') as file:
            for line in file:
                for word in line.split():
                    new_pos_stemmed.write(stem(word) + "\n")
             

        with open(os.path.join(neg_rev_dir, NEG_review), "r", encoding='UTF-8') as file:
            for line in file:
                for word in line.split():
                    new_neg_stemmed.write(stem(word) + "\n")
            

stem_all_reviews()
all_docs_full_path = list(sorted(map(lambda x: os.path.join(pos_stem_dir,x) , os.listdir(pos_stem_dir)))) + list(
    sorted(map(lambda x: os.path.join(neg_stem_dir,x), os.listdir(neg_stem_dir))))

def split_RR_NB(test_fold_id, train_test_ratio, limit):
    train, test = [], []
    pos_reviews, neg_reviews = sorted(os.listdir(pos_stem_dir)), sorted(os.listdir(neg_stem_dir))

    for index, POS_review, NEG_review in zip(range(limit), pos_reviews, neg_reviews):
        if (index % train_test_ratio) == test_fold_id:
            test.append((os.path.join(pos_stem_dir, POS_review), 'positive'))
            test.append((os.path.join(neg_stem_dir, NEG_review), 'negative'))
        else:
            train.append((os.path.join(pos_stem_dir, POS_review), 'positive'))
            train.append((os.path.join(neg_stem_dir, NEG_review), 'negative'))

    train.sort(key=operator.itemgetter(1))
    test.sort(key=operator.itemgetter(1))
    dataset = {'train': train, 'test': test}

    return dataset


# compute the frequency of unigrams in positive/negative review examples(training set)
# training set is a dictionary with pairs: (string path of review, class)
def unigram_class_count(training_set):
    pos_unigram_count, neg_unigram_count = {}, {}

    for (review,sentiment) in training_set:
        with open(review, "r", encoding='UTF-8') as review_file:
            for line in review_file:
                for word in line.split():
                    if sentiment == 'positive':
                        if word in pos_unigram_count:
                            pos_unigram_count[word] += 1
                        else:
                            pos_unigram_count[word] = 1
                    else:
                        if word in neg_unigram_count:
                            neg_unigram_count[word] += 1
                        else:
                            neg_unigram_count[word] = 1

    return {'pcount': pos_unigram_count, 'ncount': neg_unigram_count}


# compute the set of unigrams that appear at least at_least_times in the whole corpus
def get_cutoff_unigrams(at_least_times, id_feature_start):
    counts = unigram_class_count(split_RR_NB(-1, 1, len(os.listdir(pos_stem_dir)))['train'])

    unique_unigrams = {}
    feature_id = id_feature_start
    for review in all_docs_full_path:
        with open(review, "r", encoding='UTF-8') as review_file:
            for line in review_file:
                for word in line.split():
                    freq = 0
                    if word in counts['pcount']:
                        freq += counts['pcount'][word]
                    if word in counts['ncount']:
                        freq += counts['ncount'][word]
                    if word not in unique_unigrams.keys() and (freq >= at_least_times):
                        unique_unigrams[word] = feature_id
                        feature_id += 1

    return unique_unigrams


# compute the frequency of bigrams in positive/negative review examples(training set)
# training set is a dictionary with pairs: (string path of review, class)
def bigram_class_count(trainingset):
    pos_bigram_count, neg_bigram_count = {}, {}

    for (review,sentiment) in trainingset:
        with open(review, "r", encoding='UTF-8') as review_file:
            first_word = review_file.readline()
            for line in review_file:
                for word in line.split():
                    second_word = word
                    bigram = first_word + " " + second_word

                    if sentiment == 'positive':
                        if bigram in pos_bigram_count:
                            pos_bigram_count[bigram] += 1
                        else:
                            pos_bigram_count[bigram] = 1
                    else:
                        if bigram in neg_bigram_count:
                            neg_bigram_count[bigram] += 1
                        else:
                            neg_bigram_count[bigram] = 1

                    first_word = second_word

    return {'pcount': pos_bigram_count, 'ncount': neg_bigram_count}


# compute the set of bigrams that appear at least at_least_times in the whole corpus
# id_feature_start states from which integer we number the features (used for when we include bi+unigrams
def get_cutoff_bigrams(at_least_times, id_feature_start):
    counts = bigram_class_count(split_RR_NB(-1, 1, len(os.listdir(pos_stem_dir)))['train'])
    unique_bigrams = {}
    id = id_feature_start
    for review in all_docs_full_path:
        with open(review, "r", encoding='UTF-8') as review_file:
            first_word = review_file.readline()
            for line in review_file:
                for word in line.split():
                    second_word = word
                    bigram = first_word + " " + second_word
                    freq = 0
                    if bigram in counts['pcount']:
                        freq += counts['pcount'][bigram]
                    if bigram in counts['ncount']:
                        freq += counts['ncount'][bigram]
                    if bigram not in unique_bigrams.keys() and (freq >= at_least_times):
                        unique_bigrams[bigram] = id
                        id += 1
                    first_word = second_word

    return unique_bigrams

 
