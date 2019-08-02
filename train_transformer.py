from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
# from tensorboardX import SummaryWriter
# import torchvision.utils as vutils
import os
# from tqdm import tqdm
import time


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * \
        min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    if not os.path.exists("logger"):
        os.mkdir("logger")

    dataset = get_dataset()
    global_step = 0

    m = nn.DataParallel(Model().cuda())
    num_param = sum(param.numel() for param in m.parameters())
    print('Number of Transformer-TTS Parameters:', num_param)

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    # writer = SummaryWriter()

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True,
                                collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        # pbar = tqdm(dataloader)
        for i, data in enumerate(dataloader):
            # pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            character, mel, mel_input, pos_text, pos_mel, _ = data

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            # print(mel)

            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(
                character, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            loss = mel_loss + post_mel_loss

            t_l = loss.item()
            m_l = mel_loss.item()
            m_p_l = post_mel_loss.item()
            # s_l = stop_pred_loss.item()

            with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                f_total_loss.write(str(t_l)+"\n")

            with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                f_mel_loss.write(str(m_l)+"\n")

            with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                f_mel_postnet_loss.write(str(m_p_l)+"\n")

            # with open(os.path.join("logger", "stop_pred_loss.txt"), "a") as f_s_loss:
            #     f_s_loss.write(str(s_l)+"\n")

            # Print
            if global_step % hp.log_step == 0:
                # Now = time.clock()

                str1 = "Epoch [{}/{}], Step [{}], Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f};".format(
                    epoch+1, hp.epochs, global_step, mel_loss.item(), post_mel_loss.item())
                str2 = "Total Loss: {:.4f}.".format(loss.item())
                current_learning_rate = 0
                for param_group in optimizer.param_groups:
                    current_learning_rate = param_group['lr']
                str3 = "Current Learning Rate is {:.6f}.".format(
                    current_learning_rate)
                # str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                #     (Now-Start), (total_step-current_step)*np.mean(Time))

                print("\n" + str1)
                print(str2)
                print(str3)
                # print(str4)

                with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                    f_logger.write(str1 + "\n")
                    f_logger.write(str2 + "\n")
                    f_logger.write(str3 + "\n")
                    # f_logger.write(str4 + "\n")
                    f_logger.write("\n")

            # writer.add_scalars('training_loss',{
            #         'mel_loss':mel_loss,
            #         'post_mel_loss':post_mel_loss,

            #     }, global_step)

            # writer.add_scalars('alphas',{
            #         'encoder_alpha':m.module.encoder.alpha.data,
            #         'decoder_alpha':m.module.decoder.alpha.data,
            #     }, global_step)

            # if global_step % hp.image_step == 1:

            #     for i, prob in enumerate(attn_probs):

            #         num_h = prob.size(0)
            #         for j in range(4):

            #             x = vutils.make_grid(prob[j*16] * 255)
            #             writer.add_image('Attention_%d_0'%global_step, x, i*4+j)

            #     for i, prob in enumerate(attns_enc):
            #         num_h = prob.size(0)

            #         for j in range(4):

            #             x = vutils.make_grid(prob[j*16] * 255)
            #             writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)

            #     for i, prob in enumerate(attns_dec):

            #         num_h = prob.size(0)
            #         for j in range(4):

            #             x = vutils.make_grid(prob[j*16] * 255)
            #             writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model': m.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(hp.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % global_step))


if __name__ == '__main__':
    main()
