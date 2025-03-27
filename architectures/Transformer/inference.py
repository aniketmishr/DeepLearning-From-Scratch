# -*- coding: utf-8 -*-

import torch
from train import greedy_decode, get_or_build_tokenizer, get_model
from config import get_config, get_weights_file_path
import warnings
warnings.filterwarnings('ignore')

def infer_model(config ,src_text:str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # step 1: Tokenize it
    tokenizer_src = get_or_build_tokenizer(config, None, lang = config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, lang = config['lang_tgt'])
    src_token = tokenizer_src.encode(src_text).ids 
    
    # create encoder input tensor: (B, seq_len) --> (1, seq_len)
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)
    encd_num_padding = config['seq_len'] - len(src_token) - 2
    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(src_token, dtype=torch.int64), 
            eos_token, 
            torch.tensor([pad_token]*encd_num_padding, dtype=torch.int64)
        ], dim=0
    ).unsqueeze(0)
    # create encoder input mask # (1,1,seq_len)
    encoder_input_mask = (encoder_input!=pad_token).unsqueeze(0).unsqueeze(0).int()
    # create model and load the weights
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    state = torch.load("model_weight.pt", map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model_out = greedy_decode(model,encoder_input.to(device), encoder_input_mask.to(device), tokenizer_src, tokenizer_tgt, 100, device)

    # decode model output
    tgt_text = tokenizer_tgt.decode(model_out.cpu().numpy())
    return tgt_text


if __name__=='__main__':
    config = get_config()
    src_text = input("Enter: ")
    tgt_text = infer_model(config, src_text)
    print(f'Translation: {tgt_text}')
    
    with open("output.txt","w", encoding='utf-8') as f: 
        f.write(tgt_text)

    