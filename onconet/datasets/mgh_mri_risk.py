from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_mgh_mri_dataset import Abstract_MGH_MRI_Dataset

METADATA_FILENAMES = {
    5: "/home/administrator/Mounts/Isilon/metadata/mri_metadata_5years_jul30_2018.json",
    4: "/home/administrator/Mounts/Isilon/metadata/mri_metadata_4years_jun28_2018.json",
    3: "/home/administrator/Mounts/Isilon/metadata/mri_metadata_3years_jun28_2018.json",
    2: "/home/administrator/Mounts/Isilon/metadata/mri_metadata_2years_jun28_2018.json",
    1: "/home/administrator/Mounts/Isilon/metadata/mri_metadata_1years_jun28_2018.json"
}


class Abstract_MGH_MRI_Risk_Dataset(Abstract_MGH_MRI_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        self.metadata_json = METADATA_FILENAMES[self.years]
        super(Abstract_MGH_MRI_Risk_Dataset, self).__init__(args, transformer, split_group)
        self.args.num_classes = 2

    @property
    def task(self):
        return "{} Years Risk".format(self.years)

    def check_label(self, row):
        return 'label' in row and row['label'] in ['POS', 'NEG']

    def get_label(self, row):
        return row['years_to_cancer'] < self.years


@RegisterDataset("mgh_mri_5_year_risk")
class MGH_MRI_5_Year_Risk(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 5 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 5
        super(MGH_MRI_5_Year_Risk, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mri_4_year_risk")
class MGH_MRI_4_Year_Risk(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 4 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 4
        super(MGH_MRI_4_Year_Risk, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mri_3_year_risk")
class MGH_MRI_3_Year_Risk(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 3 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 3
        super(MGH_MRI_3_Year_Risk, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mri_3_year_risk_old")
class MGH_MRI_3_Year_Risk_Old(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 3 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 3.001
        super(MGH_MRI_3_Year_Risk_Old, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mri_2_year_risk")
class MGH_MRI_2_Year_Risk(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 2 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 2
        super(MGH_MRI_2_Year_Risk, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mri_1_year_risk")
class MGH_MRI_1_Year_Risk(Abstract_MGH_MRI_Risk_Dataset):
    """
    Pytorch Dataset object for the 1 year risk task on the MGH MRI dataset.
        0 - negative
        1 - positive
    on patient screening MRIs.
    """

    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_MRI_1_Year_Risk, self).__init__(args, transformer, split_group)
