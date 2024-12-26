import torch
import copy

def single2fus(model_dict, single, fus, model_fus):
    model = torch.load(model_dict)
    my_dict = copy.deepcopy(model)

    # keys_to_remove = ['embedding.embed_dict.class of worker.weight', 'embedding.embed_dict.industry code.weight', 'embedding.embed_dict.occupation code.weight', 'embedding.embed_dict.education.weight', 'embedding.embed_dict.enrolled in edu inst last wk.weight', 'embedding.embed_dict.major industry code.weight', 'embedding.embed_dict.major occupation code.weight', 'embedding.embed_dict.race.weight', 'embedding.embed_dict.hispanic origin.weight', 'embedding.embed_dict.sex.weight', 'embedding.embed_dict.member of a labor union.weight', 'embedding.embed_dict.reason for unemployment.weight', 'embedding.embed_dict.full or part time employment stat.weight', 'embedding.embed_dict.tax filer status.weight', 'embedding.embed_dict.region of previous residence.weight', 'embedding.embed_dict.state of previous residence.weight', 'embedding.embed_dict.detailed household and family stat.weight', 'embedding.embed_dict.detailed household summary in household.weight', 'embedding.embed_dict.migration code-change in msa.weight', 'embedding.embed_dict.migration code-change in reg.weight', 'embedding.embed_dict.migration code-move within reg.weight', 'embedding.embed_dict.live in this house 1 year ago.weight', 'embedding.embed_dict.migration prev res in sunbelt.weight', 'embedding.embed_dict.family members under 18.weight', 'embedding.embed_dict.country of birth father.weight', 'embedding.embed_dict.country of birth mother.weight', 'embedding.embed_dict.country of birth self.weight', 'embedding.embed_dict.citizenship.weight', 'embedding.embed_dict.own business or self employed.weight', 'embedding.embed_dict.fill inc questionnaire for veterans admin.weight', 'embedding.embed_dict.veterans benefits.weight', 'embedding.embed_dict.year.weight', 'tower.mlp.0.weight', 'tower.mlp.0.bias', 'tower.mlp.1.weight', 'tower.mlp.1.bias', 'tower.mlp.1.running_mean', 'tower.mlp.1.running_var', 'tower.mlp.1.num_batches_tracked', 'tower.mlp.4.weight', 'tower.mlp.4.bias']

    keys_to_remove = ['embedding.embed_dict.class of worker.weight', 'embedding.embed_dict.industry code.weight', 'embedding.embed_dict.occupation code.weight', 'embedding.embed_dict.education.weight', 'embedding.embed_dict.enrolled in edu inst last wk.weight', 'embedding.embed_dict.major industry code.weight', 'embedding.embed_dict.major occupation code.weight', 'embedding.embed_dict.race.weight', 'embedding.embed_dict.hispanic origin.weight', 'embedding.embed_dict.member of a labor union.weight', 'embedding.embed_dict.reason for unemployment.weight', 'embedding.embed_dict.full or part time employment stat.weight', 'embedding.embed_dict.tax filer status.weight', 'embedding.embed_dict.region of previous residence.weight', 'embedding.embed_dict.state of previous residence.weight', 'embedding.embed_dict.detailed household and family stat.weight', 'embedding.embed_dict.detailed household summary in household.weight', 'embedding.embed_dict.migration code-change in msa.weight', 'embedding.embed_dict.migration code-change in reg.weight', 'embedding.embed_dict.migration code-move within reg.weight', 'embedding.embed_dict.live in this house 1 year ago.weight', 'embedding.embed_dict.migration prev res in sunbelt.weight', 'embedding.embed_dict.family members under 18.weight', 'embedding.embed_dict.country of birth father.weight', 'embedding.embed_dict.country of birth mother.weight', 'embedding.embed_dict.country of birth self.weight', 'embedding.embed_dict.citizenship.weight', 'embedding.embed_dict.own business or self employed.weight', 'embedding.embed_dict.fill inc questionnaire for veterans admin.weight', 'embedding.embed_dict.veterans benefits.weight', 'embedding.embed_dict.year.weight', 'tower.mlp.0.weight', 'tower.mlp.0.bias', 'tower.mlp.1.weight', 'tower.mlp.1.bias', 'tower.mlp.1.running_mean', 'tower.mlp.1.running_var', 'tower.mlp.1.num_batches_tracked', 'tower.mlp.4.weight', 'tower.mlp.4.bias']

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