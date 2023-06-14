
import sklearn_crfsuite
import pycountry
from nltk.corpus import treebank
from sklearn.model_selection import cross_val_score
from sklearn_crfsuite import metrics
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import names
import geonamescache
import nltk
import csv
import sys

np.random.seed(42)

test_file = str(sys.argv[1])
prediction_file = str(sys.argv[2])

country_words = {}

for country in pycountry.countries:
    country_name = country.name.lower()
    words = country_name.split()
    country_words[country_name] = words

gc = geonamescache.GeonamesCache()
cities = gc.get_cities()
us_states = gc.get_us_states()


city_names = [city['name'].lower() for city in cities.values()]
us_states_names = [state['name'].lower() for state in us_states.values()]

#used for testing
def save_non_20_misclassificatins(crf_model, train_sentences, train_labels, train_features, valid_sentences, valid_labels, valid_features, file_name='non_20.csv'):
    def append_non_20_instances(sentences, labels, predictions, is_training, is_validation):
        non_20_instances = []
        for sentence, label_seq, prediction_seq in zip(sentences, labels, predictions):
            for (id, word, type, label), predicted_label in zip(sentence, prediction_seq):
                if label != '20' and predicted_label != label:
                    non_20_instances.append([word, predicted_label, label, is_training, is_validation])
        return non_20_instances

    y_train_pred = crf_model.predict(train_features)
    y_valid_pred = crf_model.predict(valid_features)

    non_20_instances = append_non_20_instances(train_sentences, train_labels, y_train_pred, True, False)
    non_20_instances.extend(append_non_20_instances(valid_sentences, valid_labels, y_valid_pred, False, True))

    with open(file_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Predicted label", "Actual label", "isTrainingData", "isValidationData"])  # Write the header
        writer.writerows(non_20_instances)  # Write the instances


def flatten(X):
    flattened = []
    for sentence in X:
        word = sentence[0]
        flattened+= [word]
    return flattened

def load_data(filename):
    sentences, sentence = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            if row:  
                if len(row) == 4:
                    id, word, type, label = row
                    sentence.append((id, word, type, label))
                else:  
                    id, word = row
                    sentence.append((id, word))
            else:  
                sentences.append(sentence)
                sentence = []

    if sentence:
        sentences.append(sentence)

    return sentences


def get_word_shape(word):
    shape = ""
    for char in word:
        if char == '#':
            shape += '#'
        if char.isupper():
            shape += 'X'
        elif char.islower():
            shape += 'x'
        elif char.isdigit():
            shape += 'd'
        else:
            shape += char
    return shape


def is_country(word):
    dont_match = ["the", "of", "in", "and", "see"]
    if word.lower() in country_words and word.lower() not in dont_match:
        return country_words.get(word.lower())
    else:
        return "not country"

def is_city(word):
    word = word.lower()
    if word in city_names:
        return True
    return False

def is_us_state(word):
    word = word.lower()
    if word in us_states_names:
        return True
    return False

def is_name(word):
    ambiguous_names = ["will", "rose", "way", "wait", "love", "wake", "teddy", "town", "say", "see"]
    word = word.lower()
    if word in all_names and word not in ambiguous_names:
        return True
    return False


def word2features(sentence, i):
    word = sentence[i][1]

    features = {
        'bias': 1.0,
        'word.full':word,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.iscountry()': is_country(word),
        'word.is_city()': is_city(word),
        'word.is_us_state()': is_us_state(word),
        
        'word.prefix2': word[:2],
        'word.prefix3': word[:3],
        'word.prefix4': word[:4],

        'word.suffix2': word[-2:],
        'word.suffix3': word[-3:],
        'word.suffix4': word[-4:],

        'word.capitalized': word.istitle(),
        'word.all_caps': word.isupper(),
        'word.first_char_is_num': word[0].isdigit(),
        'word.shape': get_word_shape(word),

        'word.has_hashtag': word.startswith('#'),
        'word.has_hyphen': '-' in word,
        'word_has_period': '.' in word,
        'word_has_...': '...' in word,
        'word.has_at': word.startswith('@'),
        'word.has_url': 'http' in word or 'www.' in word,
        'word.has_number': any(char.isdigit() for char in word),
    }    

    if i > 0:
        word1 = sentence[i-1][1]
        words = word1 + " " + word
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:words.lower()': words.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.iscountry()': is_country(word1),
            '-1:word.is_city()': is_city(word1),
            '-1:word.is_us_state()': is_us_state(word1),
            
            '-1:words.is_city()': is_city(words),
            '-1:words.is_us_state()': is_us_state(words),

            '-1:word.prefix2': word1[:2],
            '-1:word.prefix3': word1[:3],
            '-1:word.prefix4': word1[:4],

            '-1:word.suffix2': word1[-2:],
            '-1:word.suffix3': word1[-3:],
            '-1:word.suffix4': word1[-4:],
            
            '-1:word.has_hashtag': word1.startswith('#'),
            '-1:word.has_hyphen': '-' in word1,
            '-1:word_has_period': '.' in word1,
            '-1:word_has_...': '...' in word1,
            '-1:word.has_at': word1.startswith('@'),
            '-1:word.has_url': 'http' in word1 or 'www.' in word1,
        })
    if i > 1:
        word2 = sentence[i-2][1]
        words = word2 + " " + sentence[i-1][1] + " " + word
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:words.lower()': words.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.iscountry()': is_country(word2),
            '-2:word.is_city()': is_city(word2),
            '-2:word.is_us_state()': is_us_state(word2),
            
            '-2:words.is_city()': is_city(words),
            '-2:words.is_us_state()': is_us_state(words),

            '-2:word.prefix2': word2[:2],
            '-2:word.prefix3': word2[:3],
            '-2:word.prefix4': word2[:4],

            '-2:word.suffix2': word2[-2:],
            '-2:word.suffix3': word2[-3:],
            '-2:word.suffix4': word2[-4:],
            
            '-2:word.has_hashtag': word2.startswith('#'),
            '-2:word.has_hyphen': '-' in word2,
            '-2:word_has_period': '.' in word2,
            '-2:word_has_...': '...' in word2,
            '-2:word.has_at': word2.startswith('@'),
            '-1:word.has_url': 'http' in word1 or 'www.' in word1,
        })


    if i < len(sentence)-1:
        word1 = sentence[i+1][1]
        words = word + " " + word1
        features.update({
            '+1:words.lower()': words.lower(),
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.iscountry()': is_country(word1),
            '+1:word.is_city()': is_city(word1),
            '+1:word.is_us_state()': is_us_state(word1),

            '+1:words.is_city()': is_city(words),
            '+1:words.is_us_state()': is_us_state(words),

            '+1:word.prefix2': word1[:2],
            '+1:word.prefix3': word1[:3],
            '+1:word.prefix4': word1[:4],

            '+1:word.suffix2': word1[-2:],
            '+1:word.suffix3': word1[-3:],
            '+1:word.suffix4': word1[-4:],

            '+1:word.has_hashtag': word1.startswith('#'),
            '+1:word.has_hyphen': '-' in word1,
            '+1:word_has_period': '.' in word1,
            '+1:word_has_...': '...' in word1,
            '+1:word.has_at': word1.startswith('@'),
            '+1:word.has_url': 'http' in word1 or 'www.' in word1,
        })


    return features

def sentence2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

def sentence2labels(sentence):
    result = []
    for input in sentence:
        label = str(input[3])
        result += [str(label)]
    return result

def sentence2tokens(sentence):
    result = []
    for input in sentence:
        word = input[1]
        result += [word]
    return result

train_sents = load_data('train.csv')
valid_sents = load_data('validation.csv')
test_sents = load_data(test_file)

# shuffle training data
np.random.shuffle(train_sents)

X_train = [sentence2features(s) for s in train_sents]
y_train = [sentence2labels(s) for s in train_sents]


X_valid = [sentence2features(s) for s in valid_sents]
y_valid = [sentence2labels(s) for s in valid_sents]

X_test = [sentence2features(s) for s in test_sents]

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

df_test.to_csv(prediction_file, index=False)

y_valid_pred = crf.predict(X_valid)


labels = list(crf.classes_)
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels, zero_division=1))
labels.remove('20')
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels, zero_division=1))

save_non_20_misclassificatins(crf, train_sents, y_train, X_train, valid_sents, y_valid, X_valid, 'non_20.csv')

