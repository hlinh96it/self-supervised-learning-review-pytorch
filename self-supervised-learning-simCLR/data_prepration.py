from torch.utils.data import Dataset
from typing import Any

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
class LoadUnlabelData(Dataset):
    def __init__(self, dataset):
        super(LoadUnlabelData, self).__init__()
        self.dataset = dataset
        
    def __getitem__(self, index) -> Any:
        data = list(self.dataset[index])
        data[1] = -1
        return data
    
    def __len__(self):
        return len(self.dataset)