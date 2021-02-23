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


class SectionModel(torch.nn.Module):
    def __init__(self):
        super(SectionModel, self).__init__()
        # We need to make sure this matches the model we tokenized for!
        self.bert = AutoModel.from_pretrained('distilbert-base-cased')
        self.linear = torch.nn.Linear(768, 256, bias=False)

    def forward(self, tensor_in):
        out = self.bert(tensor_in)[0]
        out = out[:, 0, :]  # CLS token
        return self.linear(out)


def main():
    batches_per_epoch = 8192
    eval_batches_per_epoch = 1024
    save_path = Path('model.save')

    train_weighted_apikeys, test_weighted_apikeys = get_train_test_apikeys(MEMMAP_DIRECTORY)
    train_dataset = DataGenerator(MEMMAP_DIRECTORY, train_weighted_apikeys)
    train_loader = DataLoader(train_dataset, batch_size=16, pin_memory=True, num_workers=8)
    test_dataset = DataGenerator(MEMMAP_DIRECTORY, test_weighted_apikeys)
    test_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True, num_workers=12)

    model = SectionModel().cuda()
    # Diverges or just outputs the same vector for all samples at higher LRs
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    if save_path.is_file():
        print("Loading state...")
        checkpoint = torch.load(str(save_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for epoch in range(10):
        with tqdm(total=batches_per_epoch) as bar:
            bar.set_description(f"Epoch {epoch}")
            bar_loss = 0.
            model.train()
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
                loss = torch.relu(loss)  # Clip to zero
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                bar.update(1)
                bar_loss = ((bar_loss * i) + float(loss.detach())) / (i + 1)  # Rolling mean loss
                bar.set_postfix_str(f"Loss: {bar_loss:.3f}")
                if i == batches_per_epoch - 1:
                    break
        with tqdm(total=eval_batches_per_epoch) as bar:
            bar.set_description(f"Eval epoch {epoch}")
            bar_loss = 0.
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    batch = torch.reshape(batch, (-1, batch.shape[-1]))
                    outputs = model(batch)
                    outputs = torch.reshape(outputs, [-1, 3, outputs.shape[-1]])
                    positive_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 1], dim=1)
                    negative_distances = torch.linalg.norm(outputs[:, 0] - outputs[:, 2], dim=1)
                    loss = positive_distances - negative_distances + 1  # 1 is the margin term
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
        breakpoint()


if __name__ == '__main__':
    main()
