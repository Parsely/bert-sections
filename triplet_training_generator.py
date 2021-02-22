import logging
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

MAX_TOKENS_PER_DOC = 256


def get_train_test_apikeys(memmap_directory, split=0.20):
    df = pd.read_pickle(memmap_directory / 'dataframe.pkl')

    # add weights to the apikeys; these are equivalent to the sqrt of the number of posts
    unique_apikeys = pd.DataFrame(df.groupby(['apikey'])['row_number'].count().reset_index(name='num_posts'))
    unique_apikeys['weights'] = unique_apikeys[['num_posts']].apply(np.sqrt)

    # split into train and test based on apikey
    train_apikeys, test_apikeys = train_test_split(unique_apikeys, test_size=split)

    return train_apikeys, test_apikeys


def training_generator(memmap_directory, apikey_weighted_df):
    df = pd.read_pickle(memmap_directory / 'dataframe.pkl')
    data = df[df['apikey'].isin(apikey_weighted_df['apikey'])].copy(deep=True)
    word_indices = np.memmap(memmap_directory / 'word_indices.memmap', dtype=np.uint16, mode='r',
                             shape=(len(df), MAX_TOKENS_PER_DOC))
    del df
    # anchor = data_subset.copy(deep=True)
    # compare = data_subset.copy(deep=True)

    skip_count = 0
    total_count = 0

    while True:
        # sample from weighted apikey
        # We could also pregenerate the per-apikey dataframes to save time
        apikey = random.choices(apikey_weighted_df['apikey'].tolist(), k=1,
                                weights=apikey_weighted_df['weights'].tolist())[0]
        apikey_subset = data[data['apikey'] == apikey]
        # compare_subset = d[compare['apikey'] == apikey]

        # if len(compare_subset) < 2:
        #    continue
        anchor_row = apikey_subset.sample(n=1).iloc[0]
        anchor_vector = word_indices[anchor_row.row_number]
        anchor_section = anchor_row.section
        anchor_row_number = anchor_row.row_number
        total_count += 1
        try:
            positive_vector = word_indices[apikey_subset[(apikey_subset['section'] == anchor_section) & (
                        apikey_subset['row_number'] != anchor_row_number)].sample(n=1).iloc[0].row_number]
            negative = apikey_subset[(apikey_subset['section'] != anchor_section)].sample(n=1).iloc[0]
            negative_vector = word_indices[negative.row_number]
            # negative_section = negative.section
            # We store the data as np.uint16 to save space, but we definitely want a more normal
            # data type before it goes to Pytorch
            yield np.stack([anchor_vector, positive_vector, negative_vector]).astype(np.int)
            # data.append({'apikey': apikey,
            #                              'anchor': anchor_vector,
            #                              'positive': positive,
            #                              'negative': negative_text,
            #                              'section': anchor_section,
            #                              'negative_section': negative_section})
            # remove anchor row; sampling without replacement
            # anchor = anchor[anchor['row_number'] != anchor_row['row_number']]
        except ValueError:  # no positive or negative matches
            skip_count += 1
            logging.warning(f'skipped {apikey}')


def main():
    MEMMAP_DIRECTORY = Path('/Users/annelise/data/section-tripletloss-data/')
    train, _ = get_train_test_apikeys(memmap_directory=MEMMAP_DIRECTORY, split=0.20)
    batch = training_generator(MEMMAP_DIRECTORY, train)
    print(next(batch)[0].shape)


if __name__ == '__main__':
    main()
