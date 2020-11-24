import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm

METADATA_FILENAME = "florida_metadata.json" 
SUMMARY_MSG = "Constructed Florida {} {} dataset with {} records, and the following class balance \n {}"


class Abstract_Florida_Density_Dataset(Abstract_Onco_Dataset):

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        dataset = []
        class_balance = {}
        for row in tqdm.tqdm(self.metadata_json):

            def fits_criteria(row):
                group_valid = 'split_group' in row and row['split_group'] == split_group
                label_valid = self.check_label(row)
                path_valid = 'thumbnail_path' in row
                return group_valid and label_valid and path_valid

            if fits_criteria(row):
                label = self.get_label(row)
                if label not in class_balance:
                    class_balance[label] = 0
                class_balance[label] += 1
                dataset.append({
                    'path':
                    os.path.join(img_dir, row['thumbnail_path']),
                    'y':
                    label,
                    'additional': {}
                })
        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME


@RegisterDataset("florida_full_density")
class Florida_Full_Density(Abstract_Florida_Density_Dataset):
    """
    Pytorch Dataset object for the full density task on the Florida dataset.
    Full density is defined as  discriminating between:
        0- fatty
        1- scattered fibroglandular
        2- heterogeneously dense
        3- dense
    on patient mammograms.
    """

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        super(Florida_Full_Density, self).__init__(args, transformer, split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 4

    @property
    def task(self):
        return "Full Density"

    def check_label(self, row):
        return 'DENSITY' in row and row['DENSITY'] in [1, 2, 3, 4]

    def get_label(self, row):
        return row['DENSITY'] - 1


@RegisterDataset("florida_binary_density")
class Florida_Binary_Density(Abstract_Florida_Density_Dataset):
    """
    Pytorch Dataset object for the binary density task on the Florida dataset.
    Binary density is defined as  discrimanting between:
        0- low density (i.e fatty or scattered fibroglandular)
        1- high density (i.e hetrogenously dense or dense)
    on patient mammograms.
    """

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj,which can be fed in a DataLoader for batching
        '''
        super(Florida_Binary_Density, self).__init__(args, transformer,
                                                 split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Binary Density"

    def check_label(self, row):
        return 'DENSITY' in row and row['DENSITY'] in [1, 2, 3, 4]

    def get_label(self, row):
        label_map = {1: 0, 2: 0, 3: 1, 4: 1}
        return label_map[row['DENSITY']]
