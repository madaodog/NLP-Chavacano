import pandas as pd

def write_to_tsv(df, filename):
    tag_dict = {
        0: 'O',
        1: 'B-PER',
        2: 'I-PER',
        3: 'B-ORG',
        4: 'I-ORG',
        5: 'B-LOC',
        6: 'I-LOC',
    }

    with open(filename, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            for token, tag_index in zip(row['tokens'], row['ner_tags']):
                tag_label = tag_dict[tag_index]
                f.write(str(token) + '\t' + str(tag_label) + '\n')
            f.write('\n')

# Process cbk_train.parquet
train_df = pd.read_parquet('train-00000-of-00001.parquet')
write_to_tsv(train_df, 'tl_train.tsv')

# Process cbk_validation.parquet
validation_df = pd.read_parquet('validation-00000-of-00001.parquet')
write_to_tsv(validation_df, 'tl_validation.tsv')

# Process cbk_test.parquet
test_df = pd.read_parquet('test-00000-of-00001.parquet')
write_to_tsv(test_df, 'tl_test.tsv')
