from onconet.datasets.abstract_mgh_mri_dataset import Abstract_MGH_MRI_Dataset
from onconet.datasets.factory import RegisterDataset

METADATA_FILENAME = "/home/administrator/Mounts/Isilon/metadata/mri_metadata_feb11_2018_swap_augmented.json"

@RegisterDataset("mgh_mri_full_bpe")
class MGH_MRI_Full_BPE(Abstract_MGH_MRI_Dataset):
    """
    Pytorch Dataset object for the full bpe task on the MGH MRI dataset.
    Full bpe is defined as discriminating between:
        0- minimum
        1- mild
        2- moderate
        3- marked
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        self.metadata_json = METADATA_FILENAME
        super(MGH_MRI_Full_BPE, self).__init__(args, transformer, split_group)
        self.args.num_classes = 4

    def check_label(self, row):
        return 'bpe' in row and row['bpe'] in [0, 1, 2, 3]

    @property
    def task(self):
        return "Full BPE"

    def get_label(self, row):
        return row['bpe']

@RegisterDataset("mgh_mri_binary_bpe_min_vs_non_min")
class MGH_MRI_Binary_BPE(Abstract_MGH_MRI_Dataset):
    """
    Pytorch Dataset object for the binary bpe task on the MGH MRI dataset.
    Binary density is defined as  discriminating between:
        0- low bpe (i.e minimum)
        1- high bpe (i.e mild or moderate or marked)
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        self.metadata_json = METADATA_FILENAME
        super(MGH_MRI_Binary_BPE, self).__init__(args, transformer, split_group)
        self.args.num_classes = 2


    def check_label(self, row):
        return 'bpe' in row and row['bpe'] in [0, 1, 2, 3]

    @property
    def task(self):
        return "Binary BPE"

    def get_label(self, row):
        label_map = {0: 0, 1: 1, 2: 1, 3: 1}
        return label_map[row['bpe']]
