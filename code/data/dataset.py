from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self._transform = transform

    def __len__(self):
        return len(self._clusters)

    def __getitem__(self, idx):
        