import torch
import torch.nn as nn

from design_friendly.gnn.decoder import Decoder
from design_friendly.gnn.encoder import Encoder
from design_friendly.gnn.processor import Processor


class WindFarmGNN(nn.Module):
    """Overall WindFarmGNN class. Combines an encoder, a processor and a decoder to predict the node attributes
    for a given wind farm graph.

    args:
    encoder_settings: dict, the settings for the encoder
    processor_settings: dict, the settings for the processor
    decoder_settings: dict, the settings for the decoder
    norm_type: str, the type of normalization to use, one of ['mean_std', 'min_max']
    """

    def __init__(self, **kwargs):
        super().__init__()
        assert {
            "encoder_settings",
            "processor_settings",
            "decoder_settings",
            "norm_type",
        }.issubset(kwargs)
        assert kwargs["norm_type"] in ["mean_std", "min_max"]
        self.encoder = Encoder(**kwargs, **kwargs["encoder_settings"])
        self.processor = Processor(**kwargs, **kwargs["processor_settings"])
        self.decoder = Decoder(**kwargs, **kwargs["decoder_settings"])
        self.trainset_stats = None  # normalization
        self.norm_type = kwargs["norm_type"]

    def forward(self, data, denorm_output=False, return_latent=False):
        assert all(hasattr(data, attr) for attr in ["edge_index", "edge_attr", "batch"])

        def _tensors(d):  # x can be None for ambient-only; skip None
            return [
                t
                for t in (
                    getattr(d, "edge_attr", None),
                    getattr(d, "globals", None),
                    getattr(d, "x", None),
                )
                if t is not None
            ]

        assert not any(torch.isnan(t).any() for t in _tensors(data)), (
            "NaNs before forward"
        )
        data = self.normalize_input(data)
        # encode
        data.x, data.edge_attr = self.encoder(data.x, data.edge_attr, data.globals, data.batch)
        # message-passing processor to update node features of the encoded graph
        data.x = self.processor(data.x, data.edge_index, data.edge_attr)
        # decode the graph after each group to the original space
        data.x = self.decoder(data.x)
        if denorm_output:
            data = self.denormalize_output(data)
        assert not any(torch.isnan(t).any() for t in _tensors(data)), (
            "NaNs after forward"
        )
        if return_latent:
            return data, data.x
        return data

    def compute_loss(self, data):
        # loss on batch
        node_loss = nn.functional.mse_loss(
            data.x.squeeze(), data.y.squeeze(), reduction="mean"
        )
        return node_loss

    def normalize_input(self, data):
        # methd to normalize the input data using the precomputed trainset_stats
        if self.trainset_stats is None:
            raise RuntimeError("Dataset stats not initialized yet!")
        if self.norm_type == "mean_std":
            if (self.trainset_stats["x"] is not None) and (data.x is not None):
                data.x = (
                    data.x - self.trainset_stats["x"]["mean"]
                ) / self.trainset_stats["x"]["std"]
            data.y = (data.y - self.trainset_stats["y"]["mean"]) / self.trainset_stats[
                "y"
            ]["std"]
            data.globals = (
                data.globals - self.trainset_stats["globals"]["mean"]
            ) / self.trainset_stats["globals"]["std"]
            data.edge_attr = (
                data.edge_attr - self.trainset_stats["edges"]["mean"]
            ) / self.trainset_stats["edges"]["std"]

        elif self.norm_type == "min_max":
            if self.trainset_stats["x"] is not None:
                data.x = (data.x - self.trainset_stats["x"]["min"]) / (
                    self.trainset_stats["x"]["max"] - self.trainset_stats["x"]["min"]
                )
            data.y = (data.y - self.trainset_stats["y"]["min"]) / (
                self.trainset_stats["y"]["max"] - self.trainset_stats["y"]["min"]
            )
            data.globals = (data.globals - self.trainset_stats["globals"]["min"]) / (
                self.trainset_stats["globals"]["max"]
                - self.trainset_stats["globals"]["min"]
            )
            data.edge_attr = (data.edge_attr - self.trainset_stats["edges"]["min"]) / (
                self.trainset_stats["edges"]["max"]
                - self.trainset_stats["edges"]["min"]
            )

        else:
            raise NotImplementedError(
                "only mean_std or min_max normalization implemented"
            )

        return data

    def denormalize_output(self, data):
        # denormalize the data after inference using the trainset_stats
        if self.trainset_stats is None:
            raise RuntimeError("Dataset stats not initialized yet!")

        if self.norm_type == "mean_std":
            data.x = (
                data.x * self.trainset_stats["y"]["std"]
                + self.trainset_stats["y"]["mean"]
            )
            data.y = (
                data.y * self.trainset_stats["y"]["std"]
                + self.trainset_stats["y"]["mean"]
            )

        elif self.norm_type == "min_max":
            data.x = (
                data.x
                * (self.trainset_stats["y"]["max"] - self.trainset_stats["y"]["min"])
                + self.trainset_stats["y"]["min"]
            )
            data.y = (
                data.y
                * (self.trainset_stats["y"]["max"] - self.trainset_stats["y"]["min"])
                + self.trainset_stats["y"]["min"]
            )

        else:
            raise NotImplementedError(
                "only mean_std or min_max normalization implemented"
            )

        return data
