import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from network import Model
import hyperparams as hp
from text import text_to_sequence
import Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')

    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_model(num):
    model = nn.DataParallel(Model()).to(device)
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' % num))
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()
    print("Model Have Been Loaded.")

    return model


def synthesis(text, model):
    text = np.asarray(text_to_sequence(text, hp.cleaners))
    text = torch.LongTensor(text).unsqueeze(0).to(device)
    mel_input = torch.zeros([1, 1, 80]).float().to(device)
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(1000):
            pos_mel = torch.arange(
                1, mel_input.size(1)+1).unsqueeze(0).to(device)
            _, mel_postnet, _, stop_token, _, _ = model(
                text, mel_input, pos_text, pos_mel)
            if stop_token[0][-1][0].data > 0.5:
                break
            mel_input = torch.cat([mel_input, mel_postnet[:, -1:, :]], dim=1)
    print(mel_postnet.size())

    Audio.tools.inv_mel_spec(mel_postnet.transpose(1, 2), "result.wav")


if __name__ == "__main__":
    # Test
    model = get_model(6000)
    synthesis("I am very happy to see you again.", model)
