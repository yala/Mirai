import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
from tqdm import tqdm
from random import shuffle
import pickle
import numpy as np
import pdb

np.random.seed(1)


METADATA_FILENAMES = {
    'Detection' : "mammo_metadata_all_years_only_breast_cancer_nov21_2019.json"
}

ALL_VIEWS = ['L CC', 'L MLO', 'R CC','R MLO']


def all_views_present(exam):
    for key in ALL_VIEWS:
        if len(exam[key]) == 0:
            return False
    return True

SUMMARY_MSG = "Contructed MGH Mammo {} year {} {} dataset with {} records, and the following class balance \n {}"

class Abstract_MGH_Mammo_Cancer_All_Views_With_Prior_Dataset(Abstract_Onco_Dataset):

    def create_dataset(self, split_group, img_dir):
        """Gets the dataset from the paths and labels in the json.

        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
            img_dir(str): The path to the directory containing the images.

        Returns:
            The dataset, which is a list of dictionaries with each dictionary
            containing paths to the relevant images, the label, and any other
            additional information.
        """
        dataset = []
        class_balance = {}
        for mrn_row in tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']

            valid_split = split == split_group
            if self.args.use_dev_to_train_model_on_hiddens and split_group in ['train','dev']:
                if not ssn in self.args.patient_to_partition_dict:
                    self.args.patient_to_partition_dict[ssn] = np.random.choice(2)

                split_indx = 1 if split_group == 'train' else 0
                valid_split = split == 'dev' and self.args.patient_to_partition_dict[ssn] == split_indx

            if not valid_split:
                continue

            year_to_exam = {}
            for exam in exams:
                if not self.check_label(exam):
                    continue

                # Get label
                left_label = self.get_label(exam, 'L')
                right_label = self.get_label(exam, 'R')
                exam_label = self.get_label(exam, 'Any')


                if exam_label not in class_balance:
                    class_balance[exam_label] = 0

                # Determine images of left and right CCs and MLOs
                # Note: Validation of cancer side is performed in the query scripts/from_db/cancer.py in OncoQueries
                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                year = exam['sdate']

                year_to_exam[year] = {
                    'L CC': left_ccs,
                    'L MLO': left_mlos,
                    'R CC': right_ccs,
                    'R MLO': right_mlos,
                    'L y': left_label,
                    'R y': right_label,
                    'y': exam_label,
                    'exam': exam['accession']
                }

            # Go from most recent year to oldest
            all_years = list(reversed(sorted(year_to_exam.keys())))
            for indx, year in enumerate(all_years):

                exam = year_to_exam[year]
                if not all_views_present(exam):
                    continue
                prior_exams = [ (prior_year, year_to_exam[prior_year]) for prior_year in all_years[indx+1:]]

                current_paths = [exam[view][0] for view in ALL_VIEWS]

                for prior_year, prior in prior_exams:
                    if not all_views_present(prior):
                        continue

                    prior_paths = [prior[view][0] for view in ALL_VIEWS]


                    years_and_label = "cur_year:{},prior_year:{},label:{}".format(year, prior_year, exam['y'])

                    if split_group == 'train':
                        target_years = self.args.dev_years if self.args.use_dev_to_train_model_on_hiddens else self.args.train_years
                        if not (year in target_years and prior_year in target_years):
                            continue

                    elif split_group == 'dev':
                        if not (year in self.args.dev_years and
                            prior_year in self.args.dev_years):
                            continue

                    else:
                        assert split_group == 'test'
                        if not (year in self.args.test_years and
                            prior_year in self.args.test_years):
                            continue

                    if self.curr_mammos_first:
                        matched_paths = current_paths + prior_paths
                    else:
                        matched_paths = []
                        for cur_path, prior_path in  zip(current_paths, prior_paths):
                            matched_paths.append(cur_path)
                            matched_paths.append(prior_path)

                    dataset.append({
                        'paths': matched_paths,
                        'L y': exam['L y'],
                        'R y': exam['R y'],
                        'y': exam['y'],
                        'additionals': [],
                        'exam': exam['exam'],
                        'prior_exam': prior['exam'],
                        'dist_key': years_and_label,
                        'year': year,
                        'prior_year': prior_year,
                        'ssn':ssn
                    })

                    class_balance[exam['y']] += 1

        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(SUMMARY_MSG.format(self.years, self.task, split_group, len(dataset), class_balance))

        return dataset

    def check_label(self, row):
        valid_pos = row['years_to_cancer'] < self.years
        valid_neg = row['years_to_last_followup'] >= self.years
        return valid_pos or valid_neg

    def get_label(self, row, side='Any'):
        if side == 'Any':
            return row['years_to_cancer'] < self.years
        elif side == 'L':
            return row['left_years_to_cancer'] < self.years
        else:
            assert(side == 'R')
            return row['right_years_to_cancer'] < self.years

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 8

class Abstract_MGH_Mammo_Cancer_All_Views_Dataset(Abstract_MGH_Mammo_Cancer_All_Views_With_Prior_Dataset):

    def create_dataset(self, split_group, img_dir):
        """Gets the dataset from the paths and labels in the json.

        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
            img_dir(str): The path to the directory containing the images.

        Returns:
            The dataset, which is a list of dictionaries with each dictionary
            containing paths to the relevant images, the label, and any other
            additional information.
        """
        dataset = []
        class_balance = {}
        for mrn_row in tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']

            valid_split = split == split_group
            if self.args.use_dev_to_train_model_on_hiddens and split_group in ['train','dev']:
                if not ssn in self.args.patient_to_partition_dict:
                    self.args.patient_to_partition_dict[ssn] = np.random.choice(2)

                split_indx = 1 if split_group == 'train' else 0
                valid_split = split == 'dev' and self.args.patient_to_partition_dict[ssn] == split_indx

            if not valid_split:
                continue

            for exam in exams:
                if not self.check_label(exam):
                    continue

                # Get label
                left_label = self.get_label(exam, 'L')
                right_label = self.get_label(exam, 'R')
                exam_label = self.get_label(exam, 'Any')


                if exam_label not in class_balance:
                    class_balance[exam_label] = 0

                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                year = exam['sdate']
                sample = {
                    'L CC': left_ccs,
                    'L MLO': left_mlos,
                    'R CC': right_ccs,
                    'R MLO': right_mlos
                }

                years_and_label = "cur_year:{},label:{}".format(year, exam_label)

                if not all_views_present(sample):
                    continue
                current_paths = [exam[view][0] for view in ALL_VIEWS]

                if split_group == 'train':
                    target_years = self.args.dev_years if self.args.use_dev_to_train_model_on_hiddens else self.args.train_years
                    if not (year in target_years):
                        continue
                elif split_group == 'dev':
                    if not (year in self.args.dev_years):
                        continue
                else:
                    assert split_group == 'test'
                    if not (year in self.args.test_years):
                        continue

                dataset.append({
                    'paths': current_paths,
                    'L y': left_label,
                    'R y': right_label,
                    'y': exam_label,
                    'additionals': [],
                    'exam': exam['accession'],
                    'dist_key': years_and_label,
                    'year': year,
                    'ssn':ssn
                })

                class_balance[exam_label] += 1

        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(SUMMARY_MSG.format(self.years, self.task, split_group, len(dataset), class_balance))

        return dataset

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 4


@RegisterDataset("mgh_mammo_1year_detection_all_views_with_prior")
class MGH_Mammo_1Year_Detection_All_Views_With_Prior(Abstract_MGH_Mammo_Cancer_All_Views_With_Prior_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        self.curr_mammos_first = False
        self.use_prior = True
        super(MGH_Mammo_1Year_Detection_All_Views_With_Prior, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("mgh_mammo_1year_detection_all_views_with_prior_cur_first")
class MGH_Mammo_1Year_Detection_All_Views_With_Prior_Cur_First(Abstract_MGH_Mammo_Cancer_All_Views_With_Prior_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        self.curr_mammos_first = True
        self.use_prior = True
        super(MGH_Mammo_1Year_Detection_All_Views_With_Prior_Cur_First, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"


@RegisterDataset("mgh_mammo_1year_detection_all_views")
class MGH_Mammo_1Year_Detection_All_Views(Abstract_MGH_Mammo_Cancer_All_Views_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Detection_All_Views, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

