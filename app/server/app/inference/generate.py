import torch
import numpy as np
from .utils import Global, data_denormalization, delta_to_absolute

def generate_sequence(model, text, bias, prime, prime_seq, real_text,
                      hidden, window_vec, kappa, device, batch_size=1):
    if prime:
        inp = prime_seq
        real_seq = np.array(list(real_text))
        idx_arr = [Global.char_to_index.get(char) for char in real_seq]
        prime_text = np.array([idx_arr for i in range(batch_size)]).astype(np.float32)
        prime_text = torch.from_numpy(prime_text).to(device)
        prime_mask = torch.ones(prime_text.shape).to(device)
    else:
        prime_text = None
        prime_mask = None
        inp = torch.zeros(batch_size, 1, 3).to(device)

    text = np.array(list(text + "  "))

    text_idx = np.array(
        [[Global.char_to_index.get(char) for char in text] for i in range(batch_size)]
    ).astype(np.float32)

    text_tensor = torch.from_numpy(text_idx).to(device)
    text_mask = torch.ones(text_tensor.shape).to(device)

    # Generate sequence
    gen_seq = model.generate(
        inp,
        text_tensor,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vec,
        kappa,
        bias,
        prime=prime
    )

    gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    return delta_to_absolute(gen_seq[0])
