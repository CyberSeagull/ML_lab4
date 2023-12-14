import pandas as pd

toxic_df = pd.read_csv('test_probabilities_toxic.csv')
severe_toxic_df = pd.read_csv('test_probabilities_severe_toxic.csv')
obscene_df = pd.read_csv('test_probabilities_obscene.csv')
threat_df = pd.read_csv('test_probabilities_threat.csv')
insult_df = pd.read_csv('test_probabilities_insult.csv')
identity_hate_df = pd.read_csv('test_probabilities_identity_hate.csv')


combined_df = toxic_df[['id', 'toxic_probability']].copy()
combined_df['severe_toxic_probability'] = severe_toxic_df['severe_toxic_probability']
combined_df['obscene_probability'] = obscene_df['obscene_probability']
combined_df['threat_probability'] = threat_df['threat_probability']
combined_df['insult_probability'] = insult_df['insult_probability']
combined_df['identity_hate_probability'] = identity_hate_df['identity_hate_probability']

combined_df.columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

combined_df.to_csv('combined_submission.csv', index=False)
