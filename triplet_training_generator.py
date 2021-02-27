import logging
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from random import choice, sample

MAX_TOKENS_PER_DOC = 256


def get_train_test_apikeys(df, split=0.20):
    df = pd.read_pickle(df)

    # add weights to the apikeys; these are equivalent to the sqrt of the number of posts
    unique_apikeys = pd.DataFrame(df.groupby(['apikey'])['row_number'].count().reset_index(name='num_posts'))
    # unique_apikeys['weights'] = unique_apikeys[['num_posts']].apply(np.sqrt)
    unique_apikeys['weights'] = unique_apikeys[['num_posts']].astype(np.float)

    # split into train and test based on apikey
    train_apikeys, test_apikeys = train_test_split(unique_apikeys, test_size=split)

    return train_apikeys, test_apikeys


def training_generator(df, memmap, apikey_weighted_df):
    df = pd.read_pickle(df)
    data = df[df['apikey'].isin(apikey_weighted_df['apikey'])].copy(deep=True)
    word_indices = np.memmap(str(memmap), dtype=np.uint16, mode='r',
                             shape=(len(df), MAX_TOKENS_PER_DOC))
    del df
    # anchor = data_subset.copy(deep=True)
    # compare = data_subset.copy(deep=True)

    skip_count = 0
    total_count = 0

    apikey_list = list(set(data['apikey']))  # Uniquify
    # Precompute per-apikey dataframes to speed things up later
    grouped_rows = dict()
    for apikey in apikey_list:
        apikey_df = data[data['apikey'] == apikey].copy(deep=True).drop(columns=['apikey'])
        apikey_df['section'] = apikey_df['section'].cat.remove_unused_categories()
        per_apikey_sections = apikey_df.groupby(['section'])['row_number'].apply(list).reset_index()
        per_apikey_sections['weights'] = [float(len(row_number))
                                          for row_number in per_apikey_sections['row_number']]
        per_apikey_sections['weights'] = per_apikey_sections['weights'] / np.sum(per_apikey_sections['weights'])
        per_apikey_sections = per_apikey_sections.set_index('section')
        grouped_rows[apikey] = per_apikey_sections

    # We convert the weights to a probability distribution, then sample rapidly from it
    # by converting the distribution to a cumulative sum and using searchsorted
    # This gives us a random sample from O(n) weights in O(log n) time.
    apikey_choices = apikey_weighted_df['apikey'].tolist()
    apikey_weights = np.array(apikey_weighted_df['weights'].tolist())
    apikey_weights /= np.sum(apikey_weights)
    apikey_cumweights = np.cumsum(apikey_weights)

    while True:
        # Sample an apikey with the weights array
        apikey_idx = np.searchsorted(apikey_cumweights, np.random.rand(), side='left')
        apikey = apikey_choices[apikey_idx]
        apikey_subset = grouped_rows[apikey]

        # Choose a row from the sections in this apikey
        anchor_section_row = apikey_subset.sample(1, weights='weights')
        posts_in_section = anchor_section_row.iloc[0].row_number
        anchor_section = anchor_section_row.index[0]
        anchor_rownum, positive_rownum = sample(posts_in_section, 2)
        anchor_vector = word_indices[anchor_rownum]
        positive_vector = word_indices[positive_rownum]

        # Sample 2 potential negatives in case we get the same section again
        negative_sections = apikey_subset.sample(2, weights='weights').index.tolist()
        if negative_sections[0] == anchor_section:
            negative_section = negative_sections[1]
        else:
            negative_section = negative_sections[0]
        negative_rownum = choice(apikey_subset.loc[negative_section].row_number)
        negative_vector = word_indices[negative_rownum]
        # We store the data as np.uint16 to save space, but we definitely want a more normal
        # data type before it goes to Pytorch
        yield np.stack([anchor_vector, positive_vector, negative_vector]).astype(np.int)


def main():
    MEMMAP_DIRECTORY = Path('/media/data/tokenized_crawl/')
    train, _ = get_train_test_apikeys(memmap_directory=MEMMAP_DIRECTORY, split=0.20)
    batch = training_generator(MEMMAP_DIRECTORY, train)
    print(next(batch)[0].shape)


if __name__ == '__main__':
    main()
