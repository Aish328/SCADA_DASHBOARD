import torch
import torch.nn as nn


# =====================================================
# LSTM AUTOENCODER MODEL
# =====================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=9,
        hidden_size=128,
        latent_size=64,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        # =================================================
        # ENCODER
        # =================================================

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # latent compression layer
        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )

        # =================================================
        # DECODER INPUT
        # =================================================

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )

        # =================================================
        # DECODER
        # =================================================

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # =================================================
        # OUTPUT LAYER
        # =================================================

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):

        # x shape:
        # (batch_size, seq_length, input_size)

        batch_size, seq_len, _ = x.shape

        # =================================================
        # ENCODER
        # =================================================

        _, (hidden, cell) = self.encoder(x)

        # take last hidden state
        hidden_last = hidden[-1]

        # latent representation
        latent_vector = self.latent(hidden_last)

        # =================================================
        # DECODER PREPARATION
        # =================================================

        decoder_hidden = self.decoder_input(
            latent_vector
        )

        # repeat across sequence length
        decoder_input = decoder_hidden.unsqueeze(1).repeat(
            1,
            seq_len,
            1
        )

        # =================================================
        # DECODER
        # =================================================

        decoded_output, _ = self.decoder(
            decoder_input
        )

        # =================================================
        # RECONSTRUCTION
        # =================================================

        reconstructed = self.output_layer(
            decoded_output
        )

        return reconstructed