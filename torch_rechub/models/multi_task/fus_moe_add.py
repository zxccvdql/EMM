
import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class fus_moe_add(nn.Module):


    def __init__(self, features, task_types, n_level, n_expert_specific, expert_params1, expert_params2, tower_params_list):
        super().__init__()
        self.features = features
        self.n_task = len(task_types)
        self.task_types = task_types
        self.n_level = n_level
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)
        self.cgc_layers = nn.ModuleList(
            CGC(i + 1, n_level, self.n_task, n_expert_specific, self.input_dims, expert_params1, expert_params2)
            for i in range(n_level))
        self.towers = nn.ModuleList(
            MLP(expert_params1["dims"][-1], output_layer=True, **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        fus_inputs = [embed_x] * (self.n_task)
        fus_outs = []
        for i in range(self.n_level):
            fus_outs = self.cgc_layers[i](fus_inputs)  #fus_outs[i]: [batch_size, expert_dims[-1]]
            fus_inputs = fus_outs
        #predict
        ys = []
        for fus_out, tower, predict_layer in zip(fus_outs, self.towers, self.predict_layers):
            tower_out = tower(fus_out)  #[batch_size, 1]
            y = predict_layer(tower_out)  #logit -> proba
            ys.append(y)
        return torch.cat(ys, dim=1)


class CGC(nn.Module):
    
    def __init__(self, cur_level, n_level, n_task, n_expert_specific, input_dims, expert_params1, expert_params2):
        super().__init__()
        self.cur_level = cur_level  # the CGC level of fus
        self.n_level = n_level
        self.n_task = n_task
        self.n_expert_specific = n_expert_specific
        # self.n_expert_shared = n_expert_shared
        self.n_expert_all = n_expert_specific * self.n_task
        input_dims = input_dims if cur_level == 1 else expert_params1["dims"][
            -1]  #the first layer expert dim is the input data dim other expert dim
        self.experts_specific = nn.ModuleList(
            MLP(input_dims, output_layer=False, **(expert_params1 if i % 2 == 0 else expert_params2)) for i in range(self.n_task * self.n_expert_specific))
        # self.experts_shared = nn.ModuleList(
        #     MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.n_expert_shared))
        self.gates_specific = nn.ModuleList(
            MLP(
                input_dims, **{
                    "dims": [self.n_expert_specific],
                    "activation": "softmax",
                    "output_layer": False
                }) for _ in range(self.n_task))  #n_gate_specific = n_task
        # if cur_level < n_level:
        #     self.gate_shared = MLP(input_dims, **{
        #         "dims": [self.n_expert_all],
        #         "activation": "softmax",
        #         "output_layer": False
        #     })  #n_gate_specific = n_task
        self.gates_fus = nn.ModuleList(
            MLP(
                input_dims, **{
                    "dims": [self.n_task],
                    "activation": "softmax",
                    "output_layer": False
                }) for _ in range(self.n_task))
        self.att = nn.ModuleList(AttentionLayer(expert_params1["dims"][-1]) for _ in range(self.n_task))

    def forward(self, x_list):
        expert_specific_outs = []  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        for i in range(self.n_task):
            expert_specific_outs.extend([
                expert(x_list[i]).unsqueeze(1)
                for expert in self.experts_specific[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]
            ])

        # liner_out = []
        # for i in range(self.n_task):
        #     expert_outs = torch.cat(expert_specific_outs[i * self.n_expert_specific:(i + 1) * self.n_expert_specific], dim=1)
        #     liner_out.extend([gate(expert_outs).unsqueeze(1)
        #         for gate in self.gates_specific
        #     ])
        # expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared
        #                      ]  #x_list[-1]: the input for shared experts
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_specific)]  #gate_out[i]: [batch_size, n_expert_specific+n_expert_shared, 1]

        gate_fus_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_fus)] 
        
        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur_expert_list = expert_specific_outs[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]
            # cur_expert_list = expert_specific_outs
            expert_concat = torch.cat(cur_expert_list,
                                      dim=1)  #[batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_weight = torch.mul(gate_out,
                                      expert_concat)  #[batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            cgc_outs.append(expert_pooling.unsqueeze(1))
 
        att_input = torch.cat(cgc_outs, dim=1)
        out = []
        # print(att_input.shape)

        for idx, val in enumerate(gate_fus_outs):
            result = torch.zeros(att_input.shape[0], 2, att_input.shape[2], device=att_input.device)
            result[:, 0, :] = att_input[:, idx, :]
           
            gate_fus_outs_modified = val.clone()
            gate_fus_outs_modified[:, idx, :] = -float('inf')
            gate_fus_outs_max_indices = torch.argmax(gate_fus_outs_modified, dim=1)
            result[:, 1, :] = att_input[torch.arange(att_input.shape[0]).unsqueeze(1), gate_fus_outs_max_indices, :].squeeze(1)
            input_level = self.att[idx](result)
            out.append(input_level)
        return out

class AttentionLayer(nn.Module):


    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim


        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs