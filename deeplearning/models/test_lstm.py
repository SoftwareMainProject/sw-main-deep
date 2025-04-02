# test_lstm.py
import torch
import torch.nn as nn

def test_lstm():
    batch_size = 4
    seq_length = 30
    feature_dim = 256  # AutoEncoder의 latent_dim과 동일
    lstm_hidden_dim = 256
    lstm_num_layers = 1

    # 더미 시퀀스 생성: [batch_size, seq_length, feature_dim]
    dummy_seq = torch.randn(batch_size, seq_length, feature_dim)
    
    lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_dim, 
                   num_layers=lstm_num_layers, batch_first=True)
    lstm.eval()
    
    with torch.no_grad():
        lstm_out, (h_n, c_n) = lstm(dummy_seq)
    
    print("LSTM output shape:", lstm_out.shape)  # 예상: [4, 10, 256]
    print("LSTM hidden state shape:", h_n.shape)   # 예상: [1, 4, 256]

if __name__ == "__main__":
    test_lstm()
