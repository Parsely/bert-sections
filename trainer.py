from torch.utils.data import IterableDataset, DataLoader
from torch import nn
from torch.nn import functional as F
from triplet_training_generator import get_train_test_apikeys, training_generator
from pathlib import Path
from transformers import AutoModel
import torch
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

MEMMAP_DIRECTORY = Path("/media/data/tokenized_crawl")
BATCHES_PER_EPOCH = 8192


class DataGenerator(IterableDataset):
    def __init__(self, df, memmap, apikey_weighted_df):
        super(DataGenerator, self).__init__()
        self.data_generator = training_generator(df, memmap, apikey_weighted_df)

    def __iter__(self):
        return self.data_generator


class SectionModel(torch.nn.Module):
    def __init__(self, model_name):
        super(SectionModel, self).__init__()
        # We need to make sure this matches the model we tokenized for!
        # self.bert = AutoModel.from_pretrained('distilbert-base-cased')
        self.bert = AutoModel.from_pretrained(model_name)
        # self.out = torch.nn.Linear(768, 768, bias=False)

    def forward(self, tensor_in):
        out = self.bert(tensor_in)[0]
        # out = out[:, 0, :]  # CLS token
        out = out.mean(dim=1, keepdims=False)  # Mean pooling
        return out


def main(df, memmap, model_name, save_path, total_batch_size=4096):
    if 'large' in model_name:
        batch_size = 8
    else:
        batch_size = 32
    minibatches_per_update = total_batch_size // batch_size
    batches_per_epoch = (2 ** 19) // batch_size
    eval_batches_per_epoch = (2 ** 18) // batch_size

    train_weighted_apikeys, test_weighted_apikeys = get_train_test_apikeys(df)
    train_dataset = DataGenerator(df, memmap, train_weighted_apikeys)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=1)
    test_dataset = DataGenerator(df, memmap, test_weighted_apikeys)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=1)

    model = SectionModel(model_name).cuda()
    # Diverges or just outputs the same vector for all samples at higher LRs
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=1e-6)
    if save_path.is_file():
        print("Loading state...")
        checkpoint = torch.load(str(save_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    for epoch in range(start_epoch, 60):
        with tqdm(total=batches_per_epoch, dynamic_ncols=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            bar_loss = 0.
            model.eval()  # I think I don't want dropout for now
            optimizer.zero_grad()
            for i, batch in enumerate(train_loader):
                batch = batch.cuda()
                # The model expects data to have the shape (batch_size, num_tokens)
                # We reshape the data before it goes into the model to merge the batch dimension and the triplet dimension
                # Then we reconstruct those as separate dimensions afterwards
                batch = torch.reshape(batch, (-1, batch.shape[-1]))
                outputs = model(batch)
                outputs = torch.reshape(outputs, [-1, 3, outputs.shape[-1]])
                positive_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 1], dim=1)
                negative_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 2], dim=1)
                loss = positive_distances - negative_distances + 1  # 1 is the margin term
                # positive_similarities = F.cosine_similarity(outputs[:, 0], outputs[:, 1])
                # negative_similarities = F.cosine_similarity(outputs[:, 0], outputs[:, 2])
                # loss = negative_similarities - positive_similarities + 1
                loss = torch.relu(loss)
                loss = loss.mean() / minibatches_per_update
                loss.backward()
                if (i + 1) % minibatches_per_update == 0:
                    optimizer.step()
                bar.update(1)
                bar_loss = ((bar_loss * i) + float(loss.detach() * minibatches_per_update)) / (i + 1)  # Rolling mean loss
                bar.set_postfix_str(f"Loss: {bar_loss:.3f}")
                if i == batches_per_epoch - 1:
                    break
        with tqdm(total=eval_batches_per_epoch, dynamic_ncols=True) as bar:
            bar.set_description(f"Eval epoch {epoch}")
            bar_loss = 0.
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    batch = batch.cuda()
                    batch = torch.reshape(batch, (-1, batch.shape[-1]))
                    outputs = model(batch)
                    outputs = torch.reshape(outputs, [-1, 3, outputs.shape[-1]])
                    positive_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 1], dim=1)
                    negative_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 2], dim=1)
                    loss = positive_distances - negative_distances + 1  # 1 is the margin term
                    # positive_similarities = F.cosine_similarity(outputs[:, 0], outputs[:, 1])
                    # negative_similarities = F.cosine_similarity(outputs[:, 0], outputs[:, 2])
                    # loss = negative_similarities - positive_similarities + 1
                    loss = torch.relu(loss)  # Clip to zero
                    loss = loss.mean()
                    bar.update(1)
                    bar_loss = ((bar_loss * i) + float(loss.detach())) / (i + 1)  # Rolling mean loss
                    bar.set_postfix_str(f"Loss: {bar_loss:.3f}")
                    if i == eval_batches_per_epoch - 1:
                        break
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, str(save_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataframe', type=Path, required=True)
    parser.add_argument('--word_indices', type=Path, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--save_path', type=Path, required=True)
    args = parser.parse_args()
    assert args.dataframe.is_file()
    assert args.word_indices.is_file()
    main(args.dataframe, args.word_indices, args.model_name, args.save_path)
