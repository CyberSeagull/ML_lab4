import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

reduced_train = pd.read_csv('reduced_data_train.csv')
reduced_test = pd.read_csv('reduced_data_test.csv')

min_features = min(reduced_train.shape[1], reduced_test.shape[1])
reduced_train = reduced_train.iloc[:, :min_features]
reduced_test = reduced_test.iloc[:, :min_features]

init_train_set = pd.read_csv('train.csv')
init_test_set = pd.read_csv('test.csv')

train_data = init_train_set.iloc[:10000]
data_train_set = np.array(reduced_train)

test_data = init_test_set.iloc[:10000]
data_test_set = np.array(reduced_test)


def predict_and_save_probabilities(label_column,
                                   train_set,
                                   test_set,
                                   train_features,
                                   test_features,
                                   output_file):
    labels = np.array(train_set[label_column])
    classifier = RandomForestClassifier(n_estimators=120)
    classifier.fit(train_features, labels)

    # probabilities for the class
    probabilities = classifier.predict_proba(test_features)[:, 1]

    # add and save
    test_set[f'{label_column}_probability'] = probabilities
    test_set.to_csv(output_file, index=False)


train_data = init_train_set.iloc[:10000]
data_train_set = np.array(reduced_train)[:10000]

labels = ['toxic',
          'severe_toxic',
          'obscene',
          'threat',
          'insult',
          'identity_hate']

for label in labels:
    predict_and_save_probabilities(label,
                                   train_data,
                                   init_test_set,
                                   data_train_set,
                                   data_test_set,
                                   f'test_probabilities_{label}.csv')
