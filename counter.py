import pandas as pd

bert_df = pd.read_csv('electra_tagalog.txt', sep='\s+', header=None, names=['Word', 'Label'])
cbk_df = pd.read_csv('cbk_test.tsv', sep='\t', header=None, names=['Word', 'Label'])

# Merge DataFrames on their default indices
merged_df = pd.merge(bert_df, cbk_df, left_index=True, right_index=True, how='left', suffixes=('_bert', '_cbk'))


# Count the number for each possible combination of labels
count_df = merged_df.groupby(['Label_bert', 'Label_cbk']).size().reset_index(name='Count')

# Merge I's to one label
count_df['Label_bert'] = count_df['Label_bert'].apply(lambda x: x[2:] if x.startswith('I') else x)
count_df['Label_cbk'] = count_df['Label_cbk'].apply(lambda x: x[2:] if x.startswith('I') else x)

# Merge B's to one label
count_df['Label_bert'] = count_df['Label_bert'].apply(lambda x: x[2:] if x.startswith('B') else x)
count_df['Label_cbk'] = count_df['Label_cbk'].apply(lambda x: x[2:] if x.startswith('B') else x)

# Group again and sum the counts
count_df = count_df.groupby(['Label_bert', 'Label_cbk'])['Count'].sum().reset_index()

print(count_df)


