import torch


def permute_4d_5d_tensor(tensor, to_filters_last):
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return tensor
    if tensor.ndim == 4:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 1, 0))
        else:
            tensor = tensor.permute((3, 2, 0, 1))  # permute RSCK to KCRS
    elif tensor.ndim == 5:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 4, 1, 0))
        else:
            tensor = tensor.permute((4, 3, 0, 1, 2))  # permute RSTCK to KCRST
    return tensor


def permute_params(model, to_filters_last):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = permute_4d_5d_tensor(param.data, to_filters_last)


def change_state_dict_device(state_dict, to_device):
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            state_dict[name] = param.to(to_device)
    return state_dict


def adjust_tensors_for_save(state_dict, optimizer_states, to_device, to_filters_last, permute):
    if permute:
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param.data = permute_4d_5d_tensor(param.data, to_filters_last)

    change_state_dict_device(state_dict, to_device)

    for state in optimizer_states.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(to_device)

