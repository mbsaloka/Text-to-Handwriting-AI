import torch
import torch.nn as nn

import math

from .sampling import sample_from_out_dist

class HandwritingModel(nn.Module):
    def __init__(self, hidden_size=400, n_layers=3, output_size=121, vocab_size=79):
        super(HandwritingModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.K = 10  # Number of Gaussian mixtures
        self.EOS = False

        self.lstm_1 = nn.LSTM(3 + vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + vocab_size + hidden_size, hidden_size, batch_first=True)
        self.lstm_3 = nn.LSTM(3 + vocab_size + hidden_size, hidden_size, batch_first=True)

        self.window_layer = nn.Linear(hidden_size, 3 * self.K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)
        # self.init_weight()

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, self.K, 1, device=device)
        return (h0, c0), window_vector, kappa

    def one_hot_encoding(self, text):
        N, U = text.shape
        encoding = torch.zeros(N, U, self.vocab_size, device=text.device)  # Pastikan tensor di-device yang benar
        encoding.scatter_(2, text.unsqueeze(-1).long(), 1.0)  # Menggunakan scatter untuk menghindari loop
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask):
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)
        alpha, beta, kappa = mix_params.split(self.K, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(text.shape[1], dtype=torch.float32, device=text.device)
        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)

        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True

        phi = (phi * text_mask).unsqueeze(2)
        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)

        return window_vec, prev_kappa

    def init_weight(self):
        k = math.sqrt(1.0 / self.hidden_size)
        for param in self.lstm_1.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_2.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_3.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.window_layer.weight, a=-0.01, b=0.01)
        nn.init.constant_(self.window_layer.bias, 0.0)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, inputs, text, text_mask, initial_hidden, prev_window_vec, prev_kappa):
        hid_1, window_vecs = [], []
        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp_t = torch.cat((inputs[:, t:t+1], prev_window_vec), dim=2)
            hid_1_t, state_1 = self.lstm_1(inp_t, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window_vec, kappa = self.compute_window_vector(mix_params.squeeze(dim=1).unsqueeze(2), prev_kappa, text, text_mask)
            prev_window_vec = window_vec
            prev_kappa = kappa
            window_vecs.append(window_vec)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vecs = torch.cat(window_vecs, dim=1)

        inp2 = torch.cat((inputs, hid_1, window_vecs), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])
        hid_2, state_2 = self.lstm_2(inp2, state_2)

        inp3 = torch.cat((inputs, hid_2, window_vecs), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])
        hid_3, state_3 = self.lstm_3(inp3, state_3)

        concat_hid = torch.cat([hid_1, hid_2, hid_3], dim=2)
        output = self.output_layer(concat_hid) # output is y_hat

        return output, [state_1, state_2, state_3], window_vecs, prev_kappa

    def generate(self, inp, text, text_mask, prime_text, prime_mask, hidden,
                 window_vector, kappa, bias, prime=False):
        seq_len = 0
        gen_seq = []
        self.EOS = False
        with torch.no_grad():
            batch_size = inp.shape[0]
            # for adapt writing style
            if prime:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, prime_text, prime_mask, hidden, window_vector, kappa
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)

                hidden = (_hidden, _cell)
                self.EOS = False

                inp = inp.new_zeros(batch_size, 1, 3)
                _, window_vector, kappa = self.init_hidden(batch_size, inp.device)

            while not self.EOS and seq_len < 2000:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, text, text_mask, hidden, window_vector, kappa
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)

                y_hat = y_hat.squeeze()
                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        # print("Length of generated sequence:", seq_len, " | EOS:", self.EOS)
        return gen_seq
