import logging
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MEMMAP_DIRECTORY = '/Users/annelise/data/section-tripletloss-data/'
MAX_TOKENS_PER_DOC = 256


def get_train_test_apikeys(split=0.20):
    df = pd.read_pickle(MEMMAP_DIRECTORY + 'dataframe.pkl')

    # add weights to the apikeys; these are equivalent to the sqrt of the number of posts
    unique_apikeys = pd.DataFrame(df.groupby(['apikey'])['row_number'].count().reset_index(name='num_posts'))
    unique_apikeys['weights'] = unique_apikeys[['num_posts']].apply(np.sqrt)

    # split into train and test based on apikey
    train_apikeys, test_apikeys = train_test_split(unique_apikeys, test_size=split)

    return train_apikeys, test_apikeys


def training_generator(batch_size, apikey_weighted_df):
    df = pd.read_pickle(MEMMAP_DIRECTORY + 'dataframe.pkl')
    data_subset = df[df['apikey'].isin(apikey_weighted_df['apikey'])]
    word_indices = np.memmap(MEMMAP_DIRECTORY + 'word_indices.memmap', dtype=np.uint16, mode='r',
                             shape=(len(df), MAX_TOKENS_PER_DOC))
    anchor = data_subset.copy(deep=True)
    compare = data_subset.copy(deep=True)

    data = []#np.zeros((batch_size, 3, MAX_TOKENS_PER_DOC))
    skip_count = 0
    total_count = 0

    # sample from weighted apikey
    apikeys = random.choices(apikey_weighted_df['apikey'].tolist(), weights=apikey_weighted_df['weights'].tolist(),
                             k=batch_size)
    for apikey in apikeys:
        anchor_subset = anchor[anchor['apikey'] == apikey]
        compare_subset = compare[compare['apikey'] == apikey]

        if len(compare_subset) < 2:
            continue
        anchor_row = anchor_subset.sample(n=1).iloc[0]
        anchor_vector = word_indices[anchor_row.row_number]
        anchor_section = anchor_row.section
        total_count += 1
        try:
            positive_vector = word_indices[compare_subset[(compare_subset['section'] == anchor_section) & (
                        compare_subset['row_number'] != anchor_row['row_number'])].sample(n=1).iloc[0].row_number]
            negative = compare_subset[(compare_subset['section'] != anchor_section)].sample(n=1).iloc[0]
            negative_vector = word_indices[negative.row_number]
            # negative_section = negative.section
            row = [anchor_vector, positive_vector, negative_vector]
            data.append(row)
            # data.append({'apikey': apikey,
            #                              'anchor': anchor_vector,
            #                              'positive': positive,
            #                              'negative': negative_text,
            #                              'section': anchor_section,
            #                              'negative_section': negative_section})
            # remove anchor row; sampling without replacement
            anchor = anchor[anchor['row_number'] != anchor_row['row_number']]
        except ValueError:  # no positive or negative matches
            skip_count += 1
            logging.warning(f'skipped {apikey}')

    yield np.array(data)


def main():
    train, _ = get_train_test_apikeys(split=0.20)
    batch = training_generator(5, train)
    print(next(batch).shape)


if __name__ == '__main__':
    main()
