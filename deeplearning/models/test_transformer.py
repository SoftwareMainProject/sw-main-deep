# test_transformer.py
import torch
import torch.nn as nn

def test_transformer():
    batch_size = 4
    seq_length = 30
    d_model = 256  # LSTM의 hidden_dim와 일치하도록
    nhead = 4
    num_layers = 1

    # 더미 시퀀스: [batch_size, seq_length, d_model]
    dummy_seq = torch.randn(batch_size, seq_length, d_model)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    transformer_encoder.eval()
    
    with torch.no_grad():
        transformer_out = transformer_encoder(dummy_seq)
    
    print("Transformer output shape:", transformer_out.shape)  # 예상: [4, 10, 256]

if __name__ == "__main__":
    test_transformer()
