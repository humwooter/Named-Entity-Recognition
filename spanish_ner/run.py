import sklearn_crfsuite
import nltk
from nltk.corpus import cess_esp
from sklearn.model_selection import cross_val_score
from sklearn_crfsuite import metrics
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

np.random.seed(0)
nltk.download('cess_esp')


# Train POS tagger
tagger = nltk.UnigramTagger(cess_esp.tagged_sents())

def load_data(filename):
    df = pd.read_csv(filename)
    sentences, sentence = [], []    
    
    for index, row in df.iterrows():
        if pd.isnull(row).all():  # This checks if all columns in the row are NaN, indicating a blank line.
            print(row)
            if sentence:  # Only append non-empty sentences.
                sentences.append(sentence)
                sentence = []
        else:
            if 'label' in df.columns:
                sentence.append((row['id'], row['word'], row['label']))
            else:
                sentence.append((row['id'], row['word']))
    
    # After the loop, append the last sentence if it's non-empty.
    if sentence:
        sentences.append(sentence)

    return sentences


def has_accent(word):
    accents = set('áéíóúüñÁÉÍÓÚÜÑ')
    if any(char in accents for char in word):
        return True
    else:
        return False

def accent_positions(word):
    accents = set('áéíóúüñÁÉÍÓÚÜÑ')
    result = ''
    for char in word:
        if char in accents:
            result += char
        else:
            result += '_'
    return result


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

def keyword_score(word):
    # List of keywords to check for
    keywords = ['presidente', 'gobernador', 'alcalde', 'ciudad', 'país', 'continente', 'río', 'montaña', 'lago', 'océano']

    # Check if word contains any of the keywords
    has_keyword = any(keyword in word.lower() for keyword in keywords)
    is_title = word.istitle()
    is_upper = word.isupper()
    return has_keyword + is_title + is_upper

def word2features(sentence, i, wv=None):
    # print(i)
    word = sentence[i][1]
    pos_tag_word = tagger.tag([word])[0][1]
    if (pos_tag_word == None):
        pos_tag_word = "none"
    # print(pos_tag_word)

    features = {
        'bias': 1.0,
        'word.full':word,
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'word.has_number()': any(char.isdigit() for char in word),
        'word.prefix2': word[:2],
        'word.prefix3': word[:3],
        'word.prefix4': word[:4],
        'word.prefix5': word[:5],

        'word.suffix2': word[-2:],
        'word.suffix3': word[-3:],
        'word.suffix4': word[-4:],
        'word.suffix5': word[-5:],
        
        'word.accent_positions': accent_positions(word),
        'word.capitalized': word.istitle(),
        'word.all_caps': word.isupper(),
        'word.is_lower': word.islower(),
        'word.accent':has_accent(word),
        'word.keyword_score()': keyword_score(word),  # New feature to check for keywords
        'word.has_hyphen': '-' in word,
        'word.has_capital': any(char.isupper() for char in word),
        'word.pos_tag': pos_tag_word,
    }    

    

    if i > 0:
        word1 = sentence[i-1][1]
        pos_tag_word1 = tagger.tag([word1])[0][1]
        if word1.lower() == "de":
            features.update({
                '-1:word.is_de()': True,
            })
        features.update({
            '-1:word.full':word1,
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.keyword_score()': keyword_score(word1),
            '-1:word.accent':has_accent(word1),
            '-1:word.accent_positions': accent_positions(word1),

            '-1:word.prefix2': word1[:2],
            '-1:word.prefix3': word1[:3],
            '-1:word.prefix4': word1[:4],
            '-1:word.prefix5': word1[:5],
            '-1:word.suffix2': word1[-2:],
            '-1:word.suffix3': word1[-3:],
            '-1:word.suffix4': word1[-4:],
            '-1:word.suffix5': word1[-5:],

            '-1:word.pos_tag': pos_tag_word1 if pos_tag_word1 else 'none',
        })
    else:
        features['BOS'] = True

    if i < len(sentence)-1:
        word1 = sentence[i+1][1]
        pos_tag_word1 = tagger.tag([word1])[0][1]
        features.update({
            '+1:word.full':word1,
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.keyword_score()': keyword_score(word1),
            '+1:word.accent':has_accent(word1),
            '+1:word.accent_positions': accent_positions(word1),

            '+1:word.prefix2': word1[:2],
            '+1:word.prefix3': word1[:3],
            '+1:word.prefix4': word1[:4],
            '+1:word.prefix5': word1[:5],
            '+1:word.suffix2': word1[-2:],
            '+1:word.suffix3': word1[-3:],
            '+1:word.suffix4': word1[-4:],
            '+1:word.suffix5': word1[-5:],

            '+1:word.pos_tag': pos_tag_word1 if pos_tag_word1 else 'none',
        })
    else:
        features['EOS'] = True
    if i > 1: # look 2 words back
        word2 = sentence[i-2][1]
        pos_tag_word2 = tagger.tag([word2])[0][1]
        features.update({
            '-2:word.full':word2,
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.keyword_score()': keyword_score(word2),
            '-2:word.pos_tag': pos_tag_word2 if pos_tag_word2 else 'none',
            '-2:word.accent':has_accent(word2),
            '-2:word.accent_positions': accent_positions(word2),

            '-2:word.prefix2': word2[:2],
            '-2:word.prefix3': word2[:3],
            '-2:word.prefix4': word2[:4],
            '-2:word.prefix5': word2[:5],
            '-2:word.suffix2': word2[-2:],
            '-2:word.suffix3': word2[-3:],
            '-2:word.suffix4': word2[-4:],
            '-2:word.suffix5': word2[-5:],
        })

    if i < len(sentence) - 2: # look 2 words ahead
        word2 = sentence[i+2][1]
        pos_tag_word2 = tagger.tag([word2])[0][1]
        features.update({
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.full':word2,

            '+2:word.keyword_score()': keyword_score(word2),
            '+2:word.pos_tag': pos_tag_word2 if pos_tag_word2 else 'none',
            '+2:word.accent':has_accent(word2),
            '+2:word.accent_positions': accent_positions(word2),

            '+2:word.prefix2': word2[:2],
            '+2:word.prefix3': word2[:3],
            '+2:word.prefix4': word2[:4],
            '+2:word.prefix5': word2[:5],
            '+2:word.suffix2': word2[-2:],
            '+2:word.suffix3': word2[-3:],
            '+2:word.suffix4': word2[-4:],
            '+2:word.suffix5': word2[-5:],
        })

    return features

def sentence2features(sentence, wv=None):
    return [word2features(sentence, i, wv) for i in range(len(sentence))]

def sentence2labels(sentence):
    return [str(label) for id, word, label in sentence]

def sentence2tokens(sentence):
    return [word for id, word, label in sentence]

train_sents = load_data('train.csv')
valid_sents = load_data('validation.csv')
test_sents = load_data('test_noans.csv')

print(len(test_sents))
print(len(test_sents[0]))

train_sentences = [[word for id, word, label in sentence] for sentence in train_sents]

w2v_model = Word2Vec(train_sentences, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.save("word2vec.model")

wv = w2v_model.wv

X_train = [sentence2features(s, wv) for s in train_sents]
y_train = [sentence2labels(s) for s in train_sents]


X_valid = [sentence2features(s, wv) for s in valid_sents]
y_valid = [sentence2labels(s) for s in valid_sents]

X_test = [sentence2features(s, wv) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.08,
    c2=0.1,
    max_iterations=120,
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

y_valid_pred = crf.predict(X_valid)


# print report for all labels except 8
labels = list(crf.classes_)
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels))
labels.remove('8')
print(metrics.flat_f1_score(y_valid, y_val_pred, average='weighted', labels=labels))





