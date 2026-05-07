import os
import numpy as np
import torch
from collections import OrderedDict
import copy


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.
    Returns:
        meanAP (float): Mean Average Precision.
    """
    if topk == None:
        topk = database_code.shape[0]

    num_query = query_code.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    torch.cuda.empty_cache()
    return mean_AP


    
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def state_dict_to_vector(state_dict, remove_keys=[]) -> torch.Tensor:
    shared_state_dict = copy.deepcopy(state_dict)
    shared_state_dict_keys = list(shared_state_dict.keys())
    for key in remove_keys:
        for _key in shared_state_dict_keys:
            if key in _key:
                del shared_state_dict[_key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Load vector into state_dict, except the keys in `remove_keys`.
    """
    removed_keys = []
    reference_dict = copy.deepcopy(state_dict)
    reference_dict_keys = list(reference_dict.keys())
    for key in remove_keys:
        for _key in reference_dict_keys:
            if key in _key:
                removed_keys.append(_key)
                del reference_dict[_key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    return sorted_reference_dict
