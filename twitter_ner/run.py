
import sklearn_crfsuite
import nltk
from sklearn.model_selection import cross_val_score
from sklearn_crfsuite import metrics
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

np.random.seed(0)

def load_data(filename):
    df = pd.read_csv(filename)

    sentences, sentence = [], []
    last_id = -1
    for index, row in df.iterrows():
        if row['id'] < last_id:
            sentences.append(sentence)
            sentence = []
        
        if 'label' in df.columns:
            sentence.append((row['id'], row['word'], row['label']))
        else:
            sentence.append((row['id'], row['word']))
        
        last_id = row['id']
    
    if sentence:
        sentences.append(sentence)

    return sentences

def shape(word):
    if word.isdigit():
        return 'numeric'
    elif word.islower():
        return 'all_lower'
    elif word.isupper():
        return 'all_upper'
    elif word.istitle():
        return 'init_cap'
    else:
        return 'mixed_cap'

def word2features(sentence, i, postags):
    word = sentence[i][1]

    features = {
        'bias': 1.0,
        'word.full':word,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_number': any(char.isdigit() for char in word),
        'word.prefix2': word[:2],
        'word.prefix3': word[:3],
        'word.prefix4': word[:4],

        'word.suffix2': word[-2:],
        'word.suffix3': word[-3:],
        'word.suffix4': word[-4:],

        'word.capitalized': word.istitle(),
        'word.all_caps': word.isupper(),
        'word.first_char_is_num': word[0].isdigit(),
        'word.has_hyphen': '-' in word,
        'word_has_period': '.' in word,
        # 'word.all_lower': word.islower(),
        'word.shape': shape(word),
        'word.has_hashtag': word.startswith('#'),
        'word.has_at': word.startswith('@'),
        'word.has_url': 'http' in word or 'www.' in word,
        # 'word.postag': postag
    }    

    if i > 0:
        word1 = sentence[i-1][1]
        postag1 = postags[0][i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
            # '-1:word.postag': postag1
        })
    else:
        features['BOS'] = True

    if i < len(sentence)-1:
        word1 = sentence[i+1][1]
        postag1 = postags[0][i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            # '+1:word.postag': postag1
        })
    else:
        features['EOS'] = True

    return features

def sentence2features(sentence, postags):
    return [word2features(sentence, i, postags) for i in range(len(sentence))]

def sentence2labels(sentence):
    return [str(label) for id, word, label in sentence]

def sentence2tokens(sentence):
    return [word for id, word, label in sentence]

train_sents = load_data('train.csv')
valid_sents = load_data('validation.csv')
test_sents = load_data('test_noans.csv')

# print("train_sent: ", train_sents)

train_sentences = [[word for id, word, label in sentence] for sentence in train_sents]
val_sentences = [[word for id, word, label in sentence] for sentence in valid_sents]
test_sentences = [[word for id, word in sentence] for sentence in test_sents]

train_postags = [nltk.pos_tag(sentence) for sentence in train_sentences]
val_postags = [nltk.pos_tag(sentence) for sentence in val_sentences]
test_postags = [nltk.pos_tag(sentence) for sentence in test_sentences]

print("num train postags: ", len(train_postags[0]))
print("num val postags: ", len(val_sentences[0]))
print("num test postags: ", len(test_postags[0]))

# w2v_model = Word2Vec(train_sentences, vector_size=100, window=5, min_count=1, workers=4)
# w2v_model.save("word2vec.model")

# wv = w2v_model.wv

X_train = [sentence2features(s,train_postags) for s in train_sents]
y_train = [sentence2labels(s) for s in train_sents]


X_valid = [sentence2features(s,val_postags) for s in valid_sents]
y_valid = [sentence2labels(s) for s in valid_sents]

X_test = [sentence2features(s, test_postags) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_val_pred = crf.predict(X_valid)
y_test_pred = crf.predict(X_test)


results = []
for sent, pred in zip(test_sents, y_test_pred):
    for id_word_label, pred_label in zip(sent, pred):
        results.append({'id': id_word_label[0], 'label': pred_label})

df_test = pd.DataFrame(results)

df_test.to_csv('test_ans.csv', index=False)

# print report for all labels except 8
labels = list(crf.classes_)
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels))
labels.remove('20')
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels))
# y_valid_pred = crf.predict(X_valid)
# accuracy = metrics.flat_accuracy_score(y_valid, y_valid_pred)
# print('Validation accuracy:', accuracy)
