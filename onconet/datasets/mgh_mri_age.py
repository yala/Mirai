from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_mgh_mri_dataset import Abstract_MGH_MRI_Dataset

METADATA_FILENAME = "/home/administrator/Mounts/Isilon/metadata/mri_metadata_age_jul21_2018.json";


@RegisterDataset("mgh_mri_age")
class Abstract_MGH_MRI_Risk_Dataset(Abstract_MGH_MRI_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        self.metadata_json = METADATA_FILENAME
        super(Abstract_MGH_MRI_Risk_Dataset, self).__init__(args, transformer, split_group)
        self.args.num_classes = 6

    @property
    def task(self):
        return "Age"

    def check_label(self, row):
        return row['age'] != -1

    def get_label(self, row):
        cutoffs = [40, 50, 60, 70, 80, 1000] # including 1000 ensures there is always a cutoff greater than age
        cutoff = min(i for i in cutoffs if i >= row['age'])
        return cutoffs.index(cutoff)




