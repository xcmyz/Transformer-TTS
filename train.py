import torch
import torch.nn as nn

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

import network
from data_utils import TransformerTTSDataset, collate_fn, DataLoader
import hyperparams as hp


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(network.Model()).to(device)
    print("Model Ha s Been Defined")
    num_param = sum(param.numel() for param in model.parameters())
    print('Number of Transformer-TTS Parameters:', num_param)

    # Get dataset
    dataset = TransformerTTSDataset()

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    print("Defined Optimizer")

    # Get training loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=cpu_count())
    print("Got Training Loader")

    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n------Model Restored at Step %d------\n" % args.restore_step)

    except:
        print("\n------Start New Training------\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    Time = np.array(list())
    Start = time.clock()

    for epoch in range(hp.epochs):
        for i, data_of_batch in enumerate(training_loader):
            start_time = time.clock()

            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            optimizer.zero_grad()

            # Get Data
            character = torch.from_numpy(
                data_of_batch["texts"]).long().to(device)
            mel_input = torch.from_numpy(
                data_of_batch["mel_input"]).float().to(device)
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            pos_text = torch.from_numpy(
                data_of_batch["pos_text"]).long().to(device)
            pos_mel = torch.from_numpy(
                data_of_batch["pos_mel"]).long().to(device)
            stop_target = pos_mel.eq(0).float().to(device)

            # Forward
            mel_pred, postnet_pred, _, stop_preds, _, _ = model.forward(
                character, mel_input, pos_text, pos_mel)

            # Cal Loss
            mel_loss = nn.L1Loss()(mel_pred, mel_target)
            mel_postnet_loss = nn.L1Loss()(postnet_pred, mel_target)
            stop_pred_loss = nn.MSELoss()(stop_preds, stop_target)
            total_loss = mel_loss + mel_postnet_loss + stop_pred_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            m_p_l = mel_postnet_loss.item()
            s_l = stop_pred_loss.item()

            with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                f_total_loss.write(str(t_l)+"\n")

            with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                f_mel_loss.write(str(m_l)+"\n")

            with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                f_mel_postnet_loss.write(str(m_p_l)+"\n")

            with open(os.path.join("logger", "stop_pred_loss.txt"), "a") as f_s_loss:
                f_s_loss.write(str(s_l)+"\n")

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()
            current_learning_rate = adjust_learning_rate(
                optimizer, current_step)

            # Print
            if current_step % hp.log_step == 0:
                Now = time.clock()

                str1 = "Epoch [{}/{}], Step [{}/{}], Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f};".format(
                    epoch+1, hp.epochs, current_step, total_step, mel_loss.item(), mel_postnet_loss.item())
                str2 = "Stop Predicted Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    stop_pred_loss.item(), total_loss.item())
                str3 = "Current Learning Rate is {:.6f}.".format(
                    current_learning_rate)
                str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))

                print("\n" + str1)
                print(str2)
                print(str3)
                print(str4)

                with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                    f_logger.write(str1 + "\n")
                    f_logger.write(str2 + "\n")
                    f_logger.write(str3 + "\n")
                    f_logger.write(str4 + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            end_time = time.clock()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * \
        min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == "__main__":
    # Main
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
