import torch


def get_device() -> str:
    """
    Returns the optimal device for the current machine. In particular, it returns either "cuda", "mps" or "cpu", depending on the availability.
    """
    try:
        if torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
