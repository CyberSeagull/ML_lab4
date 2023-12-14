from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


train_data = pd.read_csv('cleaned_train.csv')
test_data = pd.read_csv('cleaned_test.csv')

vectorizer = CountVectorizer(max_features=5000)  # top 5000 features

X_train = vectorizer.fit_transform(train_data['clean_train'])

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train.toarray())

pd.DataFrame(X_train_pca).to_csv('reduced_data_train.csv', index=False)

batch_size = 10000
test_pca_list = []
for start in range(0, test_data.shape[0], batch_size):
    end = min(start + batch_size, test_data.shape[0])
    X_test_batch = vectorizer.transform(test_data['clean_test'][start:end])
    X_test_batch_pca = pca.transform(X_test_batch.toarray())
    test_pca_list.append(X_test_batch_pca)

X_test_pca = np.vstack(test_pca_list)

pd.DataFrame(X_test_pca).to_csv('reduced_data_test.csv', index=False)
