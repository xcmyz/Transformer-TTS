import torch
import transformer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def get_subsequent_mas_test(seq):
    # subsequent：随后的
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    # 返回上三角
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


if __name__ == "__main__":
    # Test
    # test_encoder = transformer.Models.Encoder(
    #     100, 1000, 512, 512, 2048, 6, 8, 64, 64, 0.1)
    # print(test_encoder)

    # test_enc_seq = torch.randn(2, 13).to(device)
    test_enc_seq = torch.LongTensor(2, 13).random_(0, 26)
    test_enc_pos = torch.stack(
        [torch.LongTensor([i for i in range(13)]) for i in range(2)])
    test_enc_pos = test_enc_pos.to(device)
    print(test_enc_pos.size())
    print(test_enc_pos)

    # test_enc_pos = torch.LongTensor(2, 5).random_(0, 10)

    test_encoder = transformer.Models.Encoder().to(device)
    print(test_encoder)

    out = test_encoder(test_enc_seq, test_enc_pos)
    # print(out.size())
    print(len(out))
    print(out[0].size())

    test_out = get_subsequent_mas_test(torch.randn(2, 3))
    print(test_out)

    test_decoder = transformer.Models.Decoder().to(device)
    print(test_decoder)

    test_tgt_seq = torch.stack([
        torch.Tensor([2, 3, 56, 2, 4, 5, 6, 2, 1, 1, 10, 0, 0, 0]).long(),
        torch.Tensor([2, 3, 56, 2, 4, 5, 6, 2, 1, 21, 3, 1, 0, 0]).long()
    ]).to(device)

    test_tgt_pos = torch.stack(
        [torch.LongTensor([i for i in range(14)]) for i in range(2)])

    test_mel_input = torch.randn(2, 14, 80).to(device)
    out_dec = test_decoder(test_tgt_seq, test_tgt_pos,
                           test_enc_seq, out[0], test_mel_input)
    print(out_dec[0].size())

    test_encoder_prenet = transformer.Layers.EncoderPreNet()
    print(test_encoder_prenet)

    out_e_p = test_encoder_prenet(test_enc_seq)
    print(out_e_p.size())

    test_postnet = transformer.Layers.PostNet()
    print(test_postnet)

    test_transformer = transformer.Models.TransformerTTS()
    print(test_transformer)

    mel_out, mel_out_postnet, stop_token = test_transformer(
        test_enc_seq, test_enc_pos, test_tgt_seq, test_tgt_pos, test_mel_input)

    print(mel_out.size())
    print(mel_out_postnet.size())
    print(stop_token)
