import torch
import torch.nn as nn
import math

# First step is Embedding. Turning an input sentence into a vector representation of X diemension
# Note that the original base Transformer model uses a dimension size of 512 for each word.

class InputEmbeddings(nn.Module):
    # Initializing the sizes of vocab and model dimensions, and defining the randomized embedding layer 
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # This layer maps indices (which usually represents words) to a d_model-dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, X):
        # This step aims to scale the embeddings by the square root of the diemnsion number, which is to maintain variance of the embeddings
        # It basically helps with the training stability
        return self.embedding(X) * math.sqrt(self.d_model)
    
# Second Step is positional encoding. It aims to keep track of the order of the words.
# It does so by adding more vectors to the embeddings. They have the same size of the embeddings -> (512)
# These vectors are created using a combination of sine and cosine functions of different frequencies, 
# which allows the model to learn and distinguish between different positions in a way that is smooth

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # Maximum sequence length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term) # apply cosine to odd indices

        pe = pe.unsqueeze(0) # This turn (seq_len, d_model) to (1, seq_len, d_model)

        # We use this to store pe without making it a learnable parameter
        self.register_buffer('pe', pe)
    
    def forward(self, X):
        # This function adds the positional encoding to the embeddings but it also sets requires_grad_ to False to indicate
        # that this is fixed and non-learnable
        X = X + self.pe[:, : X.shape[1], :].requires_grad_(False)
        return self.dropout(X)