import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
from tqdm import tqdm, trange
from multiprocessing import Pool, get_context
import gc
import numpy as np
from hashlib import sha1
from transformers import AutoTokenizer
from functools import partial

PARQUET_DIR = Path('/media/data/transfer_temp_delete')
OUTPUT_DIR = Path('/media/data/tokenized_crawl')
MAX_TOKENS_PER_DOC = 256

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_file(file):
    table = pq.read_table(str(file), columns=['apikey', 'title_en', 'full_content_en',
                                              'section', 'language', 'page_type'])
    # Doing it this way vastly reduces memory usage and stops leaking
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    df = df.dropna()  # Drops any row with any missing column
    # Maybe filter out too-long section names too?
    retained_rows = (
            (df['language'] == 'en')
            & (df['page_type'] == 'post')
            & (~df['apikey'].str.contains('iheart'))
            & (~df['apikey'].str.contains('medium.com'))
            & (df['full_content_en'].str.len() > 200)
            & (df['title_en'].str.len() >= 2)
            & (~df['section'].str.contains('uncategorized', case=False))
            & (~df['section'].str.contains('third party', case=False))
            & (df['section'].str.len() >= 2)
    )
    df = df[retained_rows]
    # df['content'] = df['title_en'] + '\n' + df['full_content_en']
    # df['content'] = df['content'].str.slice(0, 5000)

    df['content_hash'] = df['title_en'] + df['full_content_en']
    # Convert to 64-bit ints because built-in hash() does not reliably yield the same values
    content_hashes = [int(sha1(content.encode('utf-8')).hexdigest(), 16) % (np.iinfo('uint64').max - 10)
                      for content in df['content_hash'].tolist()]
    content_hashes = np.array(content_hashes, dtype=np.uint64)
    df['content_hash'] = content_hashes
    df = df.drop(columns=['page_type', 'language', 'title_en', 'full_content_en'])
    df['apikey'] = df['apikey'].astype('category')
    df['section'] = df['section'].astype('category')

    gc.collect()  # This seems to be weirdly essential
    return df


def process_file_to_memmap(file, identifier_df, memmap_path, memmap_shape):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    table = pq.read_table(str(file), columns=['title_en', 'full_content_en'])
    # Doing it this way vastly reduces memory usage and stops leaking
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    gc.collect()  # This seems to be weirdly essential

    df = df.dropna()

    df['content_hash'] = df['title_en'] + df['full_content_en']
    # Convert to 64-bit ints because built-in hash() does not reliably yield the same values
    content_hashes = [int(sha1(content.encode('utf-8')).hexdigest(), 16) % (np.iinfo('uint64').max - 10)
                      for content in df['content_hash'].tolist()]
    content_hashes = np.array(content_hashes, dtype=np.uint64)
    df['content_hash'] = content_hashes

    df = df.drop_duplicates(subset=['content_hash'])  # Remove duplicates, if any

    df['content'] = df['title_en'] + '\n' + df['full_content_en']
    df = df.drop(columns=['title_en', 'full_content_en'])

    df = pd.merge(df, identifier_df, how='inner', left_on='content_hash', right_index=True)

    gc.collect()
    word_indices = np.memmap(memmap_path, dtype=np.uint16, mode='r+',
                             shape=memmap_shape)
    for idx, row in df.iterrows():
        if not np.all(word_indices[row.row_number] == 0):
            continue  # This content hash has been done already
        tokens = tokenizer(row.content[:5000], return_attention_mask=False, max_length=MAX_TOKENS_PER_DOC,
                           truncation=True, return_tensors='np', padding='max_length',
                           return_token_type_ids=False)
        word_indices[row.row_number] = tokens['input_ids'][0]
    word_indices.flush()

    gc.collect()  # This seems to be weirdly essential
    return  # We just write to the memmap - nothing needs to be returned


def filter_df(df):
    df = df.drop_duplicates(subset=['content_hash'])  # Remove content duplicates, if any

    section_counts = df['section'].value_counts()
    df = pd.merge(df, section_counts, how='left', left_on='section', right_index=True)
    df['section'] = df['section_x']
    df['section_count'] = df['section_y']
    df = df.drop(columns=['section_x', 'section_y'])
    df = df[df['section_count'] >= 5]  # Sections must contain at least 5 posts

    apikey_counts = df['apikey'].value_counts()
    df = pd.merge(df, apikey_counts, how='left', left_on='apikey', right_index=True)
    df['apikey'] = df['apikey_x']
    df['apikey_count'] = df['apikey_y']
    df = df.drop(columns=['apikey_x', 'apikey_y'])
    df = df[df['apikey_count'] > 200]  # Apikeys must have at least 200 posts

    df['section_fraction'] = df['section_count'] / df['apikey_count']
    biggest_section = df.groupby('apikey').section_fraction.max()
    df = pd.merge(df, biggest_section, how='left', left_on='apikey', right_index=True)
    df['section_fraction'] = df['section_fraction_x']
    df['biggest_section'] = df['section_fraction_y']
    df = df[df['biggest_section'] < 0.3]  # No section more than 30% of posts
    df = df.drop(columns=['section_fraction_x', 'section_fraction_y', 'section_count',
                          'apikey_count', 'section_fraction', 'biggest_section'])

    sections_per_apikey = df.groupby('apikey').section.nunique()
    df = pd.merge(df, sections_per_apikey, how='left', left_on='apikey', right_index=True)
    df['section'] = df['section_x']
    df['sections_per_apikey'] = df['section_y']
    retained_rows = (df['sections_per_apikey'] < 100) & (df['sections_per_apikey'] >= 5)
    # Apikeys should have between 5 and 100 sections
    df = df[retained_rows]
    df = df.drop(columns=['section_x', 'section_y', 'sections_per_apikey'])

    df['section'] = df['section'].cat.remove_unused_categories()
    df['apikey'] = df['apikey'].cat.remove_unused_categories()

    return df


def main():
    target_files = sorted(PARQUET_DIR.glob('*.parquet'))
    # Avoids memory blowup with fork
    mp_context = get_context('forkserver')
    with mp_context.Pool(14) as p_exec:
        dataframes = list(tqdm(p_exec.imap_unordered(process_file, target_files, chunksize=1),
                               total=len(target_files), smoothing=0.))
    print("Merging and filtering...")
    df = pd.concat(dataframes, ignore_index=True)
    del dataframes
    df['section'] = df['apikey'] + '-' + df['section']
    df['section'] = df['section'].astype('category')
    df['apikey'] = df['apikey'].astype('category')
    df = filter_df(df)

    df = df.reset_index(drop=True)
    df['row_number'] = df.index
    df = df.set_index('content_hash', verify_integrity=True)

    df.to_pickle(str(OUTPUT_DIR / 'dataframe.pkl'))

    memmap_path = str(OUTPUT_DIR / 'word_indices.memmap')
    memmap_shape = (len(df), MAX_TOKENS_PER_DOC)
    word_indices = np.memmap(memmap_path, dtype=np.uint16, mode='w+',
                             shape=memmap_shape)
    word_indices.flush()  # Ensure the memmap is created
    partial_fn = partial(process_file_to_memmap, identifier_df=df,
                         memmap_path=memmap_path, memmap_shape=memmap_shape)
    for i in trange(0, len(target_files), 14):
        with mp_context.Pool(14) as p_exec:
            out = list(p_exec.imap_unordered(partial_fn, target_files[i:i+14]))
            gc.collect()


if __name__ == '__main__':
    main()
