import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
from onconet.learn.utils import BIRADS_TO_PROB
import pdb
from collections import Counter

METADATA_FILENAMES = {
    'Risk': "mammo_metadata_all_years_only_breast_cancer_nov21_2019.json",
    'Detection' : "mammo_metadata_all_years_all_cancer_sep03_2018.json"
    }

SUMMARY_MSG = "Contructed MGH Mammo {} year {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"


class Abstract_MGH_Mammo_Cancer_Dataset(Abstract_Onco_Dataset):
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []

        class_balance = {}
        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            for exam in exams:

                year = exam['sdate']

                if split_group == 'train':
                    if not (year in self.args.train_years):
                        continue
                elif split_group == 'dev':
                    if not (year in self.args.dev_years):
                        continue
                else:
                    assert split_group == 'test'
                    if not (year in self.args.test_years):
                        continue


                if self.check_label(exam):
                    left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)
                    left_label = self.get_label(exam, 'L')
                    right_label = self.get_label(exam, 'R')
                    birads = int(BIRADS_TO_PROB[exam['birads']])
                    if self.args.drop_benign_side:
                        if left_label and not right_label:
                            if len(left_ccs) + len(left_mlos) > 0:
                                right_ccs = []
                                right_mlos = []
                        if right_label and not left_label:
                            if len(right_ccs) + len(right_mlos) > 0:
                                left_ccs = []
                                left_mlos = []

                    for image_paths, label in [(left_ccs + left_mlos, left_label), (right_ccs + right_mlos, right_label)]:
                        for image_path in image_paths:
                            if label not in class_balance:
                                class_balance[label] = 0
                            class_balance[label] += 1
                            dataset.append({
                                'path': image_path,
                                'y': label,
                                'birads': birads,
                                'year': year,
                                'additional': {},
                                'exam': exam['accession'],
                                'dist_key': "{}:{}".format(year, label),
                                'ssn': ssn
                            })

        self.args.years_risk = self.years
        return dataset

    def get_summary_statement(self, dataset, split_group):
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        statement = SUMMARY_MSG.format(self.years, self.task, split_group, len(dataset), len(exams), len(patients), class_balance)
        return statement

    def check_label(self, row):
        valid_pos = row['years_to_cancer'] < self.years
        valid_neg = row['years_to_last_followup'] >= self.years
        valid_history = True if not self.task == "Risk" or self.args.use_permissive_cohort else row['years_since_cancer'] > 0
        no_cancer_now = True if not self.task == "Risk" or self.args.use_permissive_cohort else row['years_to_cancer'] > 0

        return (valid_pos or valid_neg) and valid_history and no_cancer_now

    def get_label(self, row, side='Any'):
        inv_cancer = row["years_to_invasive_cancer"] < self.years
        if side == 'Any':
            any_cancer = row["years_to_cancer"] < self.years
        elif side == 'L':
            any_cancer = row["left_years_to_cancer"] < self.years
        else:
            assert(side == 'R')
            any_cancer = row["right_years_to_cancer"] < self.years

        return any_cancer and inv_cancer if self.args.invasive_only else any_cancer
    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 2


@RegisterDataset("mgh_mammo_5year_risk")
class MGH_Mammo_5Year_Risk(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 5
        super(MGH_Mammo_5Year_Risk, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Risk"


@RegisterDataset("mgh_mammo_4year_risk")
class MGH_Mammo_4Year_Risk(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 4
        super(MGH_Mammo_4Year_Risk, self).__init__(args, transformer, split_group)
    @property
    def task(self):
        return "Risk"

@RegisterDataset("mgh_mammo_3year_risk")
class MGH_Mammo_3Year_Risk(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 3
        super(MGH_Mammo_3Year_Risk, self).__init__(args, transformer, split_group)
    @property
    def task(self):
        return "Risk"

@RegisterDataset("mgh_mammo_2year_risk")
class MGH_Mammo_2Year_Risk(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 2
        super(MGH_Mammo_2Year_Risk, self).__init__(args, transformer, split_group)
    @property
    def task(self):
        return "Risk"

@RegisterDataset("mgh_mammo_1year_risk")
class MGH_Mammo_1Year_Risk(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Risk, self).__init__(args, transformer, split_group)
    @property
    def task(self):
        return "Risk"

@RegisterDataset("mgh_mammo_5year_detection")
class MGH_Mammo_5Year_Detection(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 5
        super(MGH_Mammo_5Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_4year_detection")
class MGH_Mammo_4Year_Detection(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 4
        super(MGH_Mammo_4Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_3year_detection")
class MGH_Mammo_3Year_Detection(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 3
        super(MGH_Mammo_3Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_2year_detection")
class MGH_Mammo_2Year_Detection(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 2
        super(MGH_Mammo_2Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_1year_detection")
class MGH_Mammo_1Year_Detection(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_1year_screening_failure")
class MGH_Mammo_1Year_Screening_Failure(Abstract_MGH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Screening_Failure, self).__init__(args, transformer, split_group)

    def get_label(self, row, side='Any'):
        screening_negative = row['birads'] in ['2-Benign', '1-Negative']
        if side == 'Any':
            pos_result = row['years_to_cancer'] < self.years
        elif side == 'L':
            pos_result = row['left_years_to_cancer'] < self.years
        else:
            assert(side == 'R')
            pos_result = row['right_years_to_cancer'] < self.years

        return pos_result and screening_negative


    @property
    def task(self):
        return "Detection"

