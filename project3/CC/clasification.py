from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import pickle


categories = ['quantum physics', 'algebraic geometry', 'computer vision', 'general economics', 'statistics theory','quantitative biology']

train_data = load_files(container_path='text/train', categories=categories, shuffle=True,
                        encoding='utf-8', decode_error='replace')

# TODO - 2-1-1. Build pipeline for Naive Bayes Classifier
clf_nb = Pipeline([
    ('vect', CountVectorizer(
        stop_words='english',
        max_features= 1500,
        min_df= 4,
        max_df=0.4,
        ngram_range= (1,1)
    )),
    ('tfidf', TfidfTransformer()),
    ('clf', ComplementNB())
])
clf_nb.fit(train_data.data, train_data.target)

# TODO - 2-1-2. Build pipeline for SVM Classifier
clf_svm = Pipeline([
    ('vect', CountVectorizer(stop_words=None, max_features=2500, min_df=4, max_df=1.0, ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(C=10, kernel='linear'))
])
clf_svm.fit(train_data.data, train_data.target)

test_data = load_files(container_path='text/test', categories=categories, shuffle=True,
                        encoding='utf-8', decode_error='replace')
docs_test = test_data.data

predicted = clf_nb.predict(docs_test)
print("NB accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

predicted = clf_svm.predict(docs_test)
print("SVM accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

TEAM = 9

with open('DMA_project3_team%02d_nb.pkl' % TEAM, 'wb') as f1:
    pickle.dump(clf_nb, f1)

with open('DMA_project3_team%02d_svm.pkl' % TEAM, 'wb') as f2:
    pickle.dump(clf_svm, f2)


#NB성능 비교
'''
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np


categories = ['quantum physics', 'algebraic geometry', 'computer vision', 'general economics', 'statistics theory', 'quantitative biology']
train_data = load_files(container_path='text/train', categories=categories, shuffle=True, encoding='utf-8', decode_error='replace')
test_data = load_files(container_path='text/test', categories=categories, shuffle=True, encoding='utf-8', decode_error='replace')
docs_test = test_data.data


min_df_list = np.array([i for i in range(3, 8)])
max_df_list = [i/10 for i in range(1, 11)]
stop_word_list = {1: None, 2: 'english'}
max_feature_list = [500, 1000, 1500, 2000, 2500]
ngram_range_list = [(1, 1), (1, 2)]
classifiers = {'MultinomialNB': MultinomialNB(), 'ComplementNB': ComplementNB()}

max_accuracy = 0
max_accuracy_case = []


for clf_name, clf in classifiers.items():
    for stop_word in stop_word_list:
        for max_feature in max_feature_list:
            for min_df in min_df_list:
                for max_df in max_df_list:
                    for ngram_range in ngram_range_list:
                        count_vect = CountVectorizer(stop_words=stop_word_list[stop_word],
                                                     max_features=max_feature,
                                                     min_df=min_df, max_df=max_df,
                                                     ngram_range=ngram_range)
                        clf_pipeline = Pipeline([
                            ('vect', count_vect),
                            ('tfidf', TfidfTransformer()),
                            ('clf', clf)
                        ])
                        clf_pipeline.fit(train_data.data, train_data.target)
                        predicted = clf_pipeline.predict(docs_test)
                        accuracy = np.mean(predicted == test_data.target)
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            max_accuracy_case = [(clf_name, stop_word, max_feature, min_df, max_df, ngram_range)]
                        elif accuracy == max_accuracy:
                            max_accuracy_case.append((clf_name, stop_word, max_feature, min_df, max_df, ngram_range))


print("Max accuracy:", max_accuracy)
print("Max accuracy cases:", max_accuracy_case)
'''
#SVM 분류기와 분류기 파라미터 선택
'''
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np


categories = ['quantum physics', 'algebraic geometry', 'computer vision', 'general economics', 'statistics theory',
              'quantitative biology']
train_data = load_files(container_path='text/train', categories=categories, shuffle=True, encoding='utf-8',
                        decode_error='replace')
test_data = load_files(container_path='text/test', categories=categories, shuffle=True, encoding='utf-8',
                       decode_error='replace')
docs_test = test_data.data


classifiers = {
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'NuSVC': {
        'model': NuSVC(),
        'params': {
            'nu': [0.1, 0.5, 0.9],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'LinearSVC': {
        'model': LinearSVC(max_iter=10000),
        'params': {
            'C': [0.1, 1, 10]
        }
    }
}

max_accuracy = 0
max_accuracy_case = []


for clf_name, clf_dict in classifiers.items():
    model = clf_dict['model']
    params = clf_dict['params']


    if clf_name == 'SVC':
        for C in params['C']:
            for kernel in params['kernel']:
                for gamma in params['gamma']:
                    clf_params = {'C': C, 'kernel': kernel, 'gamma': gamma}
                    clf_pipeline = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', model.set_params(**clf_params))
                    ])
                    clf_pipeline.fit(train_data.data, train_data.target)
                    predicted = clf_pipeline.predict(docs_test)
                    accuracy = np.mean(predicted == test_data.target)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        max_accuracy_case = [(clf_name, clf_params)]
                    elif accuracy == max_accuracy:
                        max_accuracy_case.append((clf_name, clf_params))
    elif clf_name == 'NuSVC':
        for nu in params['nu']:
            for kernel in params['kernel']:
                for gamma in params['gamma']:
                    clf_params = {'nu': nu, 'kernel': kernel, 'gamma': gamma}
                    clf_pipeline = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', model.set_params(**clf_params))
                    ])
                    clf_pipeline.fit(train_data.data, train_data.target)
                    predicted = clf_pipeline.predict(docs_test)
                    accuracy = np.mean(predicted == test_data.target)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        max_accuracy_case = [(clf_name, clf_params)]
                    elif accuracy == max_accuracy:
                        max_accuracy_case.append((clf_name, clf_params))
    elif clf_name == 'LinearSVC':
        for C in params['C']:
            clf_params = {'C': C}
            clf_pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', model.set_params(**clf_params))
            ])
            clf_pipeline.fit(train_data.data, train_data.target)
            predicted = clf_pipeline.predict(docs_test)
            accuracy = np.mean(predicted == test_data.target)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_case = [(clf_name, clf_params)]
            elif accuracy == max_accuracy:
                max_accuracy_case.append((clf_name, clf_params))


print("\nMax accuracy:", max_accuracy)
print("Max accuracy cases:", max_accuracy_case)
'''
#특징 추출방법 파라미터 조정을 통한 최고의 SVM 분류기 찾기
'''
from sklearn.svm import SVC, NuSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np


categories = ['quantum physics', 'algebraic geometry', 'computer vision', 'general economics', 'statistics theory', 'quantitative biology']
train_data = load_files(container_path='text/train', categories=categories, shuffle=True, encoding='utf-8', decode_error='replace')
test_data = load_files(container_path='text/test', categories=categories, shuffle=True, encoding='utf-8', decode_error='replace')
docs_test = test_data.data


classifiers = {
    'SVC': SVC(C=10, kernel='linear'),
    'NuSVC': NuSVC(nu=0.1, kernel='linear')
}


min_df_list = np.array([i for i in range(3, 8)])
max_df_list = [i / 10 for i in range(1, 11)]
stop_word_list = {1: None, 2: 'english'}
max_feature_list = [500, 1000, 1500, 2000, 2500]
ngram_range_list = [(1, 1), (1, 2)]

max_accuracy = 0
max_accuracy_case = []
accuracies = []


total_combinations = len(classifiers) * len(stop_word_list) * len(max_feature_list) * len(min_df_list) * len(max_df_list) * len(ngram_range_list)
j = 0


for clf_name, clf in classifiers.items():
    for stop_word in stop_word_list:
        for max_feature in max_feature_list:
            for min_df in min_df_list:
                for max_df in max_df_list:
                    for ngram_range in ngram_range_list:
                        j += 1
                        count_vect = CountVectorizer(stop_words=stop_word_list[stop_word],
                                                     max_features=max_feature,
                                                     min_df=min_df, max_df=max_df,
                                                     ngram_range=ngram_range)
                        clf_pipeline = Pipeline([
                            ('vect', count_vect),
                            ('tfidf', TfidfTransformer()),
                            ('clf', clf)
                        ])
                        clf_pipeline.fit(train_data.data, train_data.target)
                        predicted = clf_pipeline.predict(docs_test)
                        accuracy = np.mean(predicted == test_data.target)
                        accuracies.append((clf_name, stop_word_list[stop_word], max_feature, min_df, max_df, ngram_range, accuracy))
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            max_accuracy_case = [(clf_name, stop_word_list[stop_word], max_feature, min_df, max_df, ngram_range)]
                        elif accuracy == max_accuracy:
                            max_accuracy_case.append((clf_name, stop_word_list[stop_word], max_feature, min_df, max_df, ngram_range))

                        
                        if j % 10 == 0:
                            print(f"{j}/{total_combinations} done")


for clf_name, stop_word, max_feature, min_df, max_df, ngram_range, accuracy in accuracies:
    print(f"Classifier: {clf_name}, Stop Words: {stop_word}, Max Features: {max_feature}, Min DF: {min_df}, Max DF: {max_df}, Ngram Range: {ngram_range}, Accuracy: {accuracy:.4f}")

print("\nMax accuracy:", max_accuracy)
print("Max accuracy cases:", max_accuracy_case)
'''