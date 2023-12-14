import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import WordNetLemmatizer
#from num2words import num2words


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

nltk.download('stopwords')
nltk.download('wordnet')

stemmer = SnowballStemmer("english")


def get_length(comment_set):
    samples_amount_comment = len(comment_set)
    return samples_amount_comment


def get_plot(comment):
    comment_length = comment.apply(len)
    plt.hist(comment_length)
    plt.title('Histogram of the length of samples')
    plt.xlabel('Length of samples')
    plt.ylabel('Frequency')
    plt.show()


def remove_symbols(comment):
    return re.sub(r'[^\w\s]','',comment)


def remove_stopwords(comment):
    stop_words = set(stopwords.words('english'))
    comment = ' '.join([word for word in comment.split() if word not in stop_words])
    return comment


def convert_lower_case(comment):
    return comment.lower()


def stemming(comment):
    max_length = 32
    comment = ' '.join([stemmer.stem(word) if len(word) <= max_length else word for word in comment.split()])
    return comment


def preprocess_data(comment):
    comment = remove_stopwords(comment)
    comment = remove_symbols(comment)
    comment = convert_lower_case(comment)
    comment = stemming(comment)
    return comment

'''
print(get_length(train_set), get_length(test_set))
get_plot(train_set['comment_text'])
get_plot(test_set['comment_text'])
'''

train_set['clean_train'] = train_set['comment_text'].apply(preprocess_data)
test_set['clean_test'] = test_set['comment_text'].apply(preprocess_data)

train_set.to_csv('cleaned_train.csv', index=False)
test_set.to_csv('cleaned_test.csv', index=False)

