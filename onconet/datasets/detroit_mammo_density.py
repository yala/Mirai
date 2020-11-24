from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from collections import Counter

METADATA_FILENAME = "/data/rsg/mammogram/detroit_data/Detroit/json_files/detroit_metadata_GE.json" #path to json file located on rosetta 10
SUMMARY_MSG = "Contructed Henry Ford Mammo {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
DENSITY_BREAKDOWN_MSG = "Number of 0: {}, number of 1: {}, number of 2: {}, number of 3: {}"

EXAM_INFORMATION_COLUMNS = {
    'density': 11,
    'png_paths': 14,
    'accession_id': 9
}

class Abstract_Detroit_Mammo_Density_Dataset(Abstract_Onco_Dataset):

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
                    if self.args.mammogram_type == "GE":
                        if "GE_MEDICAL_SYSTEMS" not in image_path:
                            continue
                    elif self.args.mammogram_type == "Hologic":
                        if "GE_MEDICAL_SYSTEMS" in image_path:
                            continue
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
                            'y': label,
                            'additional': {},
                            'exam': exam['DE_acc'], #exam id
                            'de_mrn': exam['DE_mrn'], #patient id
                            'years_to_cancer': exam['years_to_cancer']
                        })
        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        exams = set([d['exam'] for d in dataset])
        patients = set([d['de_mrn'] for d in dataset])
        densities = Counter([d['y'] for d in dataset])
        zero_density = densities[0]
        one_density = densities[1]
        two_density = densities[2]
        three_density = densities[3]

        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), len(exams), len(patients), class_balance))
        print(DENSITY_BREAKDOWN_MSG.format(zero_density, one_density, two_density, three_density))

        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

@RegisterDataset("detroit_mammo_full_density")
class Detroit_Mammo_Full_Density(Abstract_Detroit_Mammo_Density_Dataset):
    """
    Pytorch Dataset object for the full density task on the Henry Ford dataset.
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
        super(Detroit_Mammo_Full_Density, self).__init__(args, transformer, split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 4

    @property
    def task(self):
        return "Full Density"

    def check_label(self, row):
        return 'density' in row  and row['density'] in [1,2,3,4]

    def get_label(self, row):
        return row['density'] - 1


@RegisterDataset("detroit_mammo_binary_density")
class Detroit_Mammo_Binary_Density(Abstract_Detroit_Mammo_Density_Dataset):
    """
    Pytorch Dataset object for the binary density task on the Henry Ford dataset.
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
        super(Detroit_Mammo_Binary_Density, self).__init__(args, transformer,
                                                 split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Binary Density"

    def check_label(self, row):
        return 'density' in row and row['density'] in [1,2,3,4]

    def get_label(self, row):
        label_map = {1: 0, 2: 0, 3: 1, 4: 1}
        return label_map[row['density']]
