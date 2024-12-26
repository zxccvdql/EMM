import torch
import copy

def single2fus_ali(model_dict, single, fus, model_fus):
    model = torch.load(model_dict)
    my_dict = copy.deepcopy(model)

    keys_to_remove = ['embedding.embed_dict.101.weight', 'embedding.embed_dict.121.weight', 'embedding.embed_dict.122.weight', 'embedding.embed_dict.124.weight', 'embedding.embed_dict.125.weight', 'embedding.embed_dict.126.weight', 'embedding.embed_dict.127.weight', 'embedding.embed_dict.128.weight', 'embedding.embed_dict.129.weight', 'embedding.embed_dict.205.weight', 'embedding.embed_dict.206.weight', 'embedding.embed_dict.207.weight', 'embedding.embed_dict.210.weight', 'embedding.embed_dict.216.weight', 'embedding.embed_dict.508.weight', 'embedding.embed_dict.509.weight', 'embedding.embed_dict.702.weight', 'embedding.embed_dict.853.weight', 'embedding.embed_dict.301.weight', 'embedding.embed_dict.109_14.weight', 'embedding.embed_dict.110_14.weight', 'embedding.embed_dict.127_14.weight', 'embedding.embed_dict.150_14.weight', 'tower.mlp.0.weight', 'tower.mlp.0.bias', 'tower.mlp.1.weight', 'tower.mlp.1.bias', 'tower.mlp.1.running_mean', 'tower.mlp.1.running_var', 'tower.mlp.1.num_batches_tracked', 'tower.mlp.4.weight', 'tower.mlp.4.bias']
    new_dict = {k: v for k, v in my_dict.items() if k not in keys_to_remove}
    new_dict = {k.replace(single[0], fus[0]): v for k, v in new_dict.items()}
    new_dict = {k.replace(single[1], fus[1]): v for k, v in new_dict.items()}
    new_dict = {k.replace(single[2], fus[2]): v for k, v in new_dict.items()}

    keys_single = []
    for i in new_dict.keys():
        keys_single.append(i)

    model_fus.load_state_dict(new_dict, strict=False)

    for name, param in model_fus.named_parameters():
        if name in keys_single:
            param.requires_grad = False


def single2fus_ae(model_dict, single, fus, model_fus):
    model = torch.load(model_dict)
    my_dict = copy.deepcopy(model)

    keys_to_remove = ['embedding.embed_dict.categorical_1.weight', 'embedding.embed_dict.categorical_2.weight', 'embedding.embed_dict.categorical_3.weight', 'embedding.embed_dict.categorical_4.weight', 'embedding.embed_dict.categorical_5.weight', 'embedding.embed_dict.categorical_6.weight', 'embedding.embed_dict.categorical_7.weight', 'embedding.embed_dict.categorical_8.weight', 'embedding.embed_dict.categorical_9.weight', 'embedding.embed_dict.categorical_10.weight', 'embedding.embed_dict.categorical_11.weight', 'embedding.embed_dict.categorical_12.weight', 'embedding.embed_dict.categorical_13.weight', 'embedding.embed_dict.categorical_14.weight', 'embedding.embed_dict.categorical_15.weight', 'embedding.embed_dict.categorical_16.weight', 'tower.mlp.0.weight', 'tower.mlp.0.bias', 'tower.mlp.1.weight', 'tower.mlp.1.bias', 'tower.mlp.1.running_mean', 'tower.mlp.1.running_var', 'tower.mlp.1.num_batches_tracked', 'tower.mlp.4.weight', 'tower.mlp.4.bias']
    new_dict = {k: v for k, v in my_dict.items() if k not in keys_to_remove}
    new_dict = {k.replace(single[0], fus[0]): v for k, v in new_dict.items()}
    new_dict = {k.replace(single[1], fus[1]): v for k, v in new_dict.items()}
    new_dict = {k.replace(single[2], fus[2]): v for k, v in new_dict.items()}

    keys_single = []
    for i in new_dict.keys():
        keys_single.append(i)

    model_fus.load_state_dict(new_dict, strict=False)

    for name, param in model_fus.named_parameters():
        if name in keys_single:
            param.requires_grad = False