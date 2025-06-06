"""ConvLSTM_ae
    ConvLSTM with an encoding-decoding architecture
"""
from typing import Optional, Union
import sys
import argparse
import ast
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 for the 4 split in the ConvLSTM
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        cur_state: tuple
            Tuple containing the current hidden state (h) and cell state (c).

        Returns
        -------
        torch.Tensor
            Next hidden state (h_next).
        torch.Tensor
            Next cell state (c_next).
        """

        h_cur, c_cur = cur_state
        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        """
        Initialize the hidden and cell states with zeros.
        Parameters
        ----------
        batch_size: int
            Batch size.
        height: int
            Height of the input tensor.
        width: int
            Width of the input tensor.

        Returns
        -------
        torch.Tensor
            Initial hidden state (h) filled with zeros.
        torch.Tensor
            Initial cell state (c) filled with zeros.
        """
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


# SigmoidRescaler class to rescale output between -1 and 1
class SigmoidRescaler(nn.Module):
    def __init__(self):
        super(SigmoidRescaler, self).__init__()

    def forward(self, x):
        # Apply the sigmoid function
        sigmoid_output = torch.sigmoid(x)

        # Rescale the sigmoid output between -1 and 1
        rescaled_output = 2 * sigmoid_output - 1

        return rescaled_output


class ConvLSTMAE(nn.Module):
    def __init__(self, param1):
        super().__init__()

        self.hidden_dim = [64, 64, 64, 64]
        self.kernel_size = 3
        self.bias = True
        self.skip_connections = True
        self.num_inputs = 33  # 4 + 5 + 24
        self.num_outputs = 4
        self.context_length = 10
        self.target_length = 20
        self.use_weather = True

        

        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=self.num_inputs,
            hidden_dim=self.hidden_dim[0],
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=self.hidden_dim[0],
            hidden_dim=self.hidden_dim[1],
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=self.num_inputs,  # nb of s2 bands.
            hidden_dim=self.hidden_dim[0],
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=self.hidden_dim[0],
            hidden_dim=self.hidden_dim[1],
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

        padding = self.kernel_size // 2, self.kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.hidden_dim[1],
            out_channels=self.num_outputs,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=self.bias,
        )

        self.activation_output = nn.Sigmoid()  # 注意这里的输出数值范围为0到1，而不是-1到1，计算损失的时候要注意


    # @staticmethod
    # def add_model_specific_args(
    #     parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    # ):
    #     """
    #     Add model-specific arguments to the command-line argument parser.

    #     Parameters
    #     ----------
    #     parent_parser: Optional[Union[argparse.ArgumentParser, list]]
    #         Parent argument parser (optional).

    #     Returns
    #     -------
    #     argparse.ArgumentParser
    #         Argument parser with added model-specific arguments.
    #     """
    #     # Create a new argument parser or use the parent parser
    #     if parent_parser is None:
    #         parent_parser = []
    #     elif not isinstance(parent_parser, list):
    #         parent_parser = [parent_parser]

    #     parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

    #     # Add model-specific arguments
    #     parser.add_argument(
    #         "--hidden_dim", type=ast.literal_eval, default=[64, 64, 64, 64]
    #     )
    #     parser.add_argument("--kernel_size", type=int, default=3)
    #     parser.add_argument("--bias", type=str2bool, default=True)
    #     # parser.add_argument("--setting", type=str, default="en22")
    #     parser.add_argument("--num_inputs", type=int, default=4 + 5 + 24)
    #     parser.add_argument("--num_outputs", type=int, default=4)
    #     parser.add_argument("--context_length", type=int, default=10)
    #     parser.add_argument("--target_length", type=int, default=20)
    #     parser.add_argument("--skip_connections", type=str2bool, default=True)
    #     parser.add_argument("--teacher_forcing", type=str2bool, default=False)
    #     parser.add_argument("--target", type=str, default=False)
    #     parser.add_argument("--use_weather", type=str2bool, default=True)
    #     # parser.add_argument("--spatial_shuffle", type = str2bool, default = False)
    #     return parser

    def forward(
        self,
        data
    ):
        """
        Forward pass of the ConvLSTMAE model.

        Parameters
        ----------
        data: dict
            Dictionary containing the input data.
        pred_start: int, optional
            Starting index of predictions (default is 0).
        preds_length: Optional[int], optional
            Length of predictions (default is None).
        step: Optional[int], optional
            Step parameter for teacher forcing (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor containing the model predictions.
        dict
            Empty dictionary as the second return value (not used in this implementation).
        """

        # Determine the context length for prediction
        context_length = self.context_length

        # Extract data components

        # sentinel 2 bands
        sentinel = data["imgs"][:, :context_length, ...]  # B, T, 4, H, W

        weather = data["meso_condition_image"].unsqueeze(3).unsqueeze(4)  # B, T, 24, 1, 1

        static = data["highres_condition_image"]  # B, 5, H, W

        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        b, t, _, h, w = sentinel.shape


        # Initialize hidden states for encoder ConvLSTM cells
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, height=h, width=w)
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, height=h, width=w
        )

        output = []

        # Encoding network
        for t in range(context_length):
            # Prepare input for encoder ConvLSTM cells
            input = torch.cat((sentinel[:, t, ...], static), dim=1)  # B, 9, H, W
            # WARNING only for En23
            
            if self.use_weather:
                weather_t = (
                    weather[:, t, ...]
                    .view(weather.shape[0], 1, -1, 1, 1)
                    .squeeze(1)
                    .repeat(1, 1, 128, 128)
                )  # B, 24, H, W
                input = torch.cat((input, weather_t), dim=1)  # B, 33, H, W

            # First ConvLSTM block
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input, cur_state=[h_t, c_t])
            # Second ConvLSTM block
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

        # First prediction
        pred = self.conv(h_t2)

        # Add the last frame of the context period if skip_connections is True
        if self.skip_connections:
            pred = pred + sentinel[:, t, ...]  # B, 4, H, W

        pred = self.activation_output(pred)

        # Forecasting network
        for t in range(self.target_length):
            # Copy the previous prediction for skip connections
            pred_previous = torch.clone(pred)

            pred = torch.cat((pred, static), dim=1)
            if self.use_weather:
                # Prepare input for decoder ConvLSTM cells
                weather_t = (
                    weather[:, context_length + t, ...]
                    .view(weather.shape[0], 1, -1, 1, 1)
                    .squeeze(1)
                    .repeat(1, 1, 128, 128)
                )
                pred = torch.cat((pred, weather_t), dim=1)  # B, 33, H, W

            # First ConvLSTM block for the decoder
            h_t, c_t = self.decoder_1_convlstm(input_tensor=pred, cur_state=[h_t, c_t])

            # Second ConvLSTM block for the decoder
            h_t2, c_t2 = self.decoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )

            pred = h_t2
            pred = self.conv(pred)

            # Add the previous prediction for skip connections
            if self.skip_connections:
                pred = pred + pred_previous

            # Output
            pred = self.activation_output(pred)
            output += [pred.unsqueeze(1)]

        output = torch.cat(output, dim=1)  # B,20,4,H,W

        return output
