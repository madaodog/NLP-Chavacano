import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('0000.parquet')

text_data = df
tokens = text_data['tokens']
tags = text_data['ner_tags']


train_df, temp_df = train_test_split(df, test_size=0.15, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

tag_dict = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
}

def write_to_tsv(df, filename):
    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            for token, tag_index in zip(row['tokens'], row['ner_tags']):
                tag_label = tag_dict[tag_index]
                f.write(str(token) + '\t' + str(tag_label) + '\n')
            f.write('\n')  

write_to_tsv(train_df, 'cbk_train.tsv')
write_to_tsv(dev_df, 'cbk_dev.tsv')
write_to_tsv(test_df, 'cbk_test.tsv')