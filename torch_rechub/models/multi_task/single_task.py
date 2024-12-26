"""
Date: create on 12/05/2022
References: 
    paper: (AKDD'2017) Deep & Cross Network for Ad Click Predictions
    url: https://arxiv.org/abs/1708.05123
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class single_task(torch.nn.Module):
    """Deep & Cross Network

    Args:
        features (list[Feature Class]): training by the whole module.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, features, task_types, hidden_units, tower_params_list):
        super().__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])

        self.embedding = EmbeddingLayer(features)
        # self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp1 = MLP(self.dims, output_layer=False, **hidden_units)
        self.mlp2 = MLP(hidden_units["dims"][-1], output_layer=False, **hidden_units)
        self.mlp3 = MLP(hidden_units["dims"][-1], output_layer=False, **hidden_units)
        # self.linear = LR(self.dims + mlp_params["dims"][-1])
        self.tower = MLP(hidden_units["dims"][-1], output_layer=True, **tower_params_list)
        self.predict_layer = PredictionLayer(task_types[0]) 

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        mlp_out = self.mlp1(embed_x)
        mlp_out = self.mlp2(mlp_out)
        mlp_out = self.mlp2(mlp_out)

        tower_out = self.tower(mlp_out)
        y = self.predict_layer(tower_out)
        return y