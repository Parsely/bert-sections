from torch.utils.data import IterableDataset, DataLoader
from triplet_training_generator import get_train_test_apikeys, training_generator
from pathlib import Path
from transformers import AutoModel
import torch
from tqdm import tqdm

MEMMAP_DIRECTORY = Path("/media/data/tokenized_crawl")
BATCHES_PER_EPOCH = 8192


class DataGenerator(IterableDataset):
    def __init__(self, memmap_directory, apikey_weighted_df):
        super(DataGenerator, self).__init__()
        self.data_generator = training_generator(memmap_directory, apikey_weighted_df)

    def __iter__(self):
        return self.data_generator


def main():
    train_weighted_apikeys, test_weighted_apikeys = get_train_test_apikeys(MEMMAP_DIRECTORY)
    train_dataset = DataGenerator(MEMMAP_DIRECTORY, train_weighted_apikeys)
    train_loader = DataLoader(train_dataset, batch_size=16, pin_memory=True, num_workers=8)
    test_dataset = DataGenerator(MEMMAP_DIRECTORY, test_weighted_apikeys)
    test_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True)
    # We need to make sure this matches the model we tokenized for!
    model = AutoModel.from_pretrained('distilbert-base-cased').cuda()
    with tqdm(total=8192) as bar:
        bar.set_description("Epoch 0")
        bar_loss = 0.
        for i, batch in enumerate(train_loader):
            batch = batch.cuda()
            # The model expects data to have the shape (batch_size, num_tokens)
            # We reshape the data before it goes into the model to merge the batch dimension and the triplet dimension
            # Then we reconstruct those as separate dimensions afterwards
            batch = torch.reshape(batch, (-1, batch.shape[-1]))
            outputs = model(batch)[0]
            outputs = outputs[:, 0, :]  # We read the output from the CLS token only
            outputs = torch.reshape(outputs, [-1, 3, outputs.shape[-1]])
            positive_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 1], dim=1)
            negative_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 2], dim=1)
            loss = negative_distances - positive_distances + 1  # 1 is the margin term
            loss = loss.mean()
            loss.backward()
            bar.update(1)
            bar_loss = ((bar_loss * i) + float(loss.detach())) / (i + 1)  # Rolling mean loss
            bar.set_postfix_str(f"Loss: {bar_loss:.3f}")

if __name__ == '__main__':
    main()