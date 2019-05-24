import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerTTSLoss(nn.Module):
    """ TransformerTTS Loss """

    def __init__(self):
        super(TransformerTTSLoss, self).__init__()

    def forward(self, mel_output, mel_output_postnet, gate_predicted, mel_target, gate_target):

        mel_target.requires_grad = False
        mel_loss = torch.abs(mel_output - mel_target)
        mel_loss = torch.mean(mel_loss)

        mel_postnet_loss = torch.abs(mel_output_postnet - mel_target)
        mel_postnet_loss = torch.mean(mel_postnet_loss)

        gate_loss = nn.BCEWithLogitsLoss()(gate_predicted, gate_target)

        return mel_loss, mel_postnet_loss, gate_loss
