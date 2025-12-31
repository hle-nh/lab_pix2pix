import torch
def get_fixed_samples(dataset, num_samples=5):
    indices = torch.linspace(0, len(dataset)-1, num_samples).long()
    samples = [dataset[i] for i in indices]
    Ls, abs_ = zip(*samples)
    return torch.stack(Ls), torch.stack(abs_)
