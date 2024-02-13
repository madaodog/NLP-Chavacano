import pandas as pd
import math

df = pd.read_parquet('0000.parquet')

text_data = df
tokens = text_data['tokens']
tags = text_data['ner_tags']



train_df_length = math.ceil(0.7 * len(df))
dev_df_length = math.ceil(0.85 * len(df))
test_df_length = len(df)

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

write_to_tsv(df[:train_df_length], 'cbk_train.tsv')
write_to_tsv(df[train_df_length:dev_df_length], 'cbk_dev.tsv')
write_to_tsv(df[dev_df_length:test_df_length], 'cbk_test.tsv')