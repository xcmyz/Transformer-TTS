import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from multiprocessing import cpu_count
import os

import hyperparams as hp
from text import text_to_sequence


class TransformerTTSDataset(Dataset):
    """ LJSpeech """

    def __init__(self, dataset_path=hp.data_path):
        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.text_path)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index)
        mel_target = np.load(mel_name)
        mel_input = np.concatenate(
            [np.zeros([1, hp.num_mels], np.float32), mel_target[:-1, :]], axis=0)

        character = self.text[idx]
        character = text_to_sequence(character, hp.cleaners)
        character = np.array(character)

        return {"text": character, "mel_target": mel_target, "mel_input": mel_input}


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def collate_fn(batch):
    texts = [d['text'] for d in batch]
    mel_targets = [d['mel_target'] for d in batch]
    mel_inputs = [d['mel_input'] for d in batch]

    texts, pos_text = pad_text(texts)
    mel_inputs, pos_mel = pad_mel(mel_inputs)
    mel_targets, _ = pad_mel(mel_targets)

    out = {"texts": texts, "pos_text": pos_text, "mel_target": mel_targets,
           "pos_mel": pos_mel, "mel_input": mel_inputs}
    return out


def pad_text(inputs):

    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=pad)
        pos_padded = np.pad(np.array([(i+1) for i in range(np.shape(x)[0])]),
                            (0, length - x.shape[0]), mode='constant', constant_values=pad)

        return x_padded, pos_padded

    max_len = max((len(x) for x in inputs))

    text_padded = np.stack([pad_data(x, max_len)[0] for x in inputs])
    pos_padded = np.stack([pad_data(x, max_len)[1] for x in inputs])

    return text_padded, pos_padded


def pad_mel(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)
        pos_padded = np.pad(np.array([(i+1) for i in range(np.shape(x)[0])]),
                            (0, max_len - x.shape[0]), mode='constant', constant_values=pad)

        return x[:, :s], pos_padded

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len)[0] for x in inputs])
    pos_padded = np.stack([pad(x, max_len)[1] for x in inputs])

    return mel_output, pos_padded


if __name__ == "__main__":
    # Test
    test_dataset = TransformerTTSDataset()
    test_loader = DataLoader(test_dataset, batch_size=1,
                             collate_fn=collate_fn, drop_last=True, num_workers=1)
    for i, batch in enumerate(test_loader):
        print(i)
