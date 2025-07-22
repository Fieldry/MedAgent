import os

import lightning as L
import pandas as pd
import torch
import torch.utils.data as data


class EhrDataset(data.Dataset):
    def __init__(self, data_path, task, mode="train"):
        super().__init__()
        self.dataset = pd.read_pickle(os.path.join(data_path, f"{mode}_data.pkl"))
        self.id = [item["id"] for item in self.dataset]
        self.data = [item["x_ts"] for item in self.dataset]
        self.label = [item[f"y_{task}"] for item in self.dataset]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.id[index]


class EhrDataModule(L.LightningDataModule):
    def __init__(self, data_path, task, batch_size=32, test_mode: str="test", train_sample_weights=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = EhrDataset(data_path, task, mode="train")
        self.val_dataset = EhrDataset(data_path, task, mode="val")
        self.test_dataset = EhrDataset(data_path, task, mode=test_mode)
        self.train_sample_weights = train_sample_weights

    def train_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        if self.train_sample_weights is not None:
            sampler = data.WeightedRandomSampler(
                weights=self.train_sample_weights,
                num_samples=len(self.train_sample_weights),
                replacement=True
            )
            return data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=self.pad_collate, num_workers=8, multiprocessing_context=multiprocessing_context, persistent_workers=True)
        else:
            return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.pad_collate, num_workers=8, multiprocessing_context=multiprocessing_context, persistent_workers=True)

    def train_dataloader_for_update(self):
        """
        A special dataloader for getting predictions on the full training set
        in its original order, which is needed for updating weights.
        """
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.pad_collate, num_workers=8, multiprocessing_context=multiprocessing_context)

    def val_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    def test_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8, persistent_workers=True, multiprocessing_context=multiprocessing_context)

    def pad_collate(self, batch):
        xx, yy, pid = zip(*batch)
        lens = torch.as_tensor([len(x) for x in xx])
        # convert to tensor
        xx = [torch.tensor(x) for x in xx]
        yy = [torch.tensor(y) for y in yy]
        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
        return xx_pad.float(), yy_pad.float(), lens, pid