import pandas as pd

from emotechintentionmodelv3 import emounifiedintentionmodel


def prep_synonyms(csv_file):
    data = pd.read_csv(csv_file)

    data['Sentence'] = data['Sentence'].str.strip()
    data['Response'] = data['Response'].str.strip()

    if 'intents' not in data:
        data['Response'] = data['Response'].str.replace('\n', '\\n')
        data['Response'] = data['Response'].str.replace('\t', '\\t')
        data['Response'] = data['Response'].str.replace('_', '\_')
        data['Response'] = data['Response'].str.replace('‚Äô', '\'')
        data['Sentence'] = data['Sentence'].str.replace('‚Äô', '\'')
        data['Response'] = data['Response'].str.replace("'", '\'')
        data['Sentence'] = data['Sentence'].str.replace("''", '\'')
        data['intents'], data['objects'], data['synonyms'], data['ner'], data['emotion'] = [list(_) for _ in zip(
            *data['Sentence'].apply(lambda x: emounifiedintentionmodel(x, 5)))]
    data.reset_index(inplace=True, drop=True)
    data.to_csv(csv_file, index=True, index_label="Index")
