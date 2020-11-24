from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm

METADATA_FILENAME = "mammo_metadata_dec31_2017.json"
SUMMARY_MSG = "Contructed MGH Mammo {} {} dataset with {} records, and the following class balance \n {}"


class Abstract_MGH_Mammo_Density_Dataset(Abstract_Onco_Dataset):

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        dataset = []
        class_balance = {}
        for mrn_row in tqdm.tqdm(self.metadata_json):
            split, exams = mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            for exam in exams:
                for image_path in exam['png_paths']:
                    def fits_criteria(exam):
                        label_valid = self.check_label(exam)
                        return label_valid

                    if fits_criteria(exam):
                        label = self.get_label(exam)
                        if label not in class_balance:
                            class_balance[label] = 0
                        class_balance[label] += 1
                        dataset.append({
                            'path':
                            image_path,
                            'y':
                            label,
                            'additional': {},
                            'exam':
                            exam['accession']
                        })
        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME


@RegisterDataset("mgh_mammo_full_density")
class MGH_Mammo_Full_Density(Abstract_MGH_Mammo_Density_Dataset):
    """
    Pytorch Dataset object for the full density task on the MGH dataset.
    Full density is defined as  discrimanting between:
        0- fatty
        1- cattered fibroglandular
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
        super(MGH_Mammo_Full_Density, self).__init__(args, transformer, split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 4

    @property
    def task(self):
        return "Full Density"

    def check_label(self, row):
        return 'density' in row and row['density'] in [1, 2, 3, 4]

    def get_label(self, row):
        return row['density'] - 1


@RegisterDataset("mgh_mammo_binary_density")
class MGH_Mammo_Binary_Density(Abstract_MGH_Mammo_Density_Dataset):
    """
    Pytorch Dataset object for the binary density task on the MGH dataset.
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
        super(MGH_Mammo_Binary_Density, self).__init__(args, transformer,
                                                 split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Binary Density"

    def check_label(self, row):
        return 'density' in row and row['density'] in [1, 2, 3, 4]

    def get_label(self, row):
        label_map = {1: 0, 2: 0, 3: 1, 4: 1}
        return label_map[row['density']]
