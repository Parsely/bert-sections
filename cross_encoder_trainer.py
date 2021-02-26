from torch.utils.data import IterableDataset, DataLoader
from torch import nn
from torch.nn import functional as F
from triplet_training_generator import get_train_test_apikeys, training_generator
from pathlib import Path
from transformers import AutoModel
import torch
from tqdm import tqdm
import pandas as pd

MEMMAP_DIRECTORY = Path("/media/data/tokenized_crawl")
BATCHES_PER_EPOCH = 8192


class DataGenerator(IterableDataset):
    def __init__(self, memmap_directory, apikey_weighted_df):
        super(DataGenerator, self).__init__()
        self.data_generator = training_generator(memmap_directory, apikey_weighted_df)

    def __iter__(self):
        return self.data_generator


class CrossEncoderModel(torch.nn.Module):
    def __init__(self):
        super(CrossEncoderModel, self).__init__()
        # We need to make sure this matches the model we tokenized for!
        # self.bert = AutoModel.from_pretrained('distilbert-base-cased')
        self.bert = AutoModel.from_pretrained('distilbert-base-cased')
        self.hidden = nn.Linear(768, 512)
        self.out = nn.Linear(512, 1)
        # self.out = torch.nn.Linear(768, 768, bias=False)

    def forward(self, tensor_in, sep_token_id=102):
        positive_pairs = torch.cat([tensor_in[:, 0], tensor_in[:, 1]], dim=1)
        positive_pairs[:, 256] = sep_token_id
        negative_pairs = torch.cat([tensor_in[:, 0], tensor_in[:, 2]], dim=1)
        negative_pairs[:, 256] = sep_token_id
        positive_labels = torch.ones(len(positive_pairs), dtype=torch.float32, device=tensor_in.device)
        negative_labels = torch.zeros_like(positive_labels)
        labels = torch.cat([positive_labels, negative_labels])
        inputs = torch.cat([positive_pairs, negative_pairs], dim=0)
        assert len(labels) == inputs.shape[0]
        out = self.bert(inputs)[0]
        # out = out[:, 0, :]  # CLS token
        out = out.mean(dim=1, keepdims=False)  # Mean pooling
        out = F.gelu(self.hidden(out))
        out = torch.squeeze(self.out(out))
        loss = F.binary_cross_entropy_with_logits(out, labels)
        return loss


def main():
    batch_size = 16
    batches_per_epoch = (2 ** 19) // batch_size
    eval_batches_per_epoch = (2 ** 18) // batch_size
    save_path = Path('model.save')

    train_weighted_apikeys, test_weighted_apikeys = get_train_test_apikeys(MEMMAP_DIRECTORY)
    debug_weighted_apikeys = pd.concat([train_weighted_apikeys, test_weighted_apikeys]).query('num_posts > 1000000')
    train_dataset = DataGenerator(MEMMAP_DIRECTORY, debug_weighted_apikeys)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=1)
    test_dataset = DataGenerator(MEMMAP_DIRECTORY, debug_weighted_apikeys)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=1)

    model = CrossEncoderModel().cuda()
    # Diverges or just outputs the same vector for all samples at higher LRs
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=1e-4)
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
            model.train()
            optimizer.zero_grad()
            for i, batch in enumerate(train_loader):
                batch = batch.cuda()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                bar.update(1)
                bar_loss = ((bar_loss * i) + float(loss.detach())) / (i + 1)  # Rolling mean loss
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
                    loss = model(batch)
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
    main()
