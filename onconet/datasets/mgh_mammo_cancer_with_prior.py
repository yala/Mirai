from tqdm import tqdm
from collections import OrderedDict, Counter
import numpy as np
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.factory import RegisterDataset
from onconet.utils.generic import normalize_dictionary
import pdb

SUMMARY_MSG = "Contructed MGH Mammo {} year {} {} dataset with {} records, and the following class balance \n {}"

METADATA_FILENAMES = {
    'Risk': "mammo_metadata_all_years_only_breast_cancer_aug04_2018.json",
    'Detection' : "mammo_metadata_all_years_all_cancer_aug04_2018.json"
}

class Abstract_MGH_Mammo_Cancer_With_Prior_Dataset(Abstract_Onco_Dataset):
    """A risk dataset where each input consists of an mammo view, and a prior of that same view. Concretly, suppose a patient has 3 mammograms, each with a Left CC/MLO and a Right CC MLO, for a total of 12 images. This dataset would create 12 dataset samples. 4 comparing the most recent set of views to their immediate prior, 4 comparing them to the older prior, and 4 comparing the immediate and older prior."""

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
            if not split == split_group:
                continue

            year_to_exam = {}
            for exam in exams:
                if not self.check_label(exam):
                    continue

                # Get label
                left_label = self.get_label(exam, 'L')
                right_label = self.get_label(exam, 'R')

                for label in left_label, right_label:
                    if label not in class_balance:
                        class_balance[label] = 0

                # Determine images of left and right CCs and MLOs
                # Note: Validation of cancer side is performed in the query scripts/from_db/cancer.py in OncoQueries
                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                if self.args.drop_benign_side:
                    if left_label and not right_label:
                        right_ccs = []
                        right_mlos = []
                    if right_label and not left_label:
                        left_ccs = []
                        left_mlos = []

                year = exam['sdate']

                year_to_exam[year] = {
                    'L CC': left_ccs,
                    'L MLO': left_mlos,
                    'R CC': right_ccs,
                    'R MLO': right_mlos,
                    'L y': left_label,
                    'R y': right_label,
                    'exam': exam['accession']
                }

            # Go from most recent year to oldest
            all_views = ['L CC', 'L MLO', 'R CC','R MLO']
            all_years = list(reversed(sorted(year_to_exam.keys())))
            for indx, year in enumerate(all_years):

                exam = year_to_exam[year]
                prior_exams = [ (prior_year, year_to_exam[prior_year]) for prior_year in all_years[indx+1:]]
                for view in all_views:
                    side = view.split()[0]

                    if len(exam[view]) == 0:
                        continue

                    prior_with_view = [ (prior_year, prior) for prior_year, prior in prior_exams if len(prior[view]) > 0]

                    if len(prior_with_view) == 0:
                        continue

                    for prior_year, prior in prior_with_view:
                        years_and_label = "cur_year:{},prior_year:{},label:{}".format(year, prior_year, exam['{} y'.format(side)])

                        if split_group == 'train':
                            if not (year in self.args.train_years and prior_year in self.args.train_years):
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

                        dataset.append({
                            'paths': [exam[view][0], prior[view][0]],
                            'y': exam['{} y'.format(side)],
                            'additionals': [],
                            'exam': exam['exam'],
                            'prior_exam': prior['exam'],
                            'dist_key': years_and_label,
                            'year': year,
                            'prior_year': prior_year,
                            'ssn':ssn
                        })

                        class_balance[exam['{} y'.format(side)]] += 1

        class_balance = normalize_dictionary(class_balance)
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
        args.num_images = 2



@RegisterDataset("mgh_mammo_5year_risk_with_prior")
class MGH_Mammo_5Year_Risk_With_Prior(Abstract_MGH_Mammo_Cancer_With_Prior_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 5
        super(MGH_Mammo_5Year_Risk_With_Prior, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Risk"
@RegisterDataset("mgh_mammo_1year_risk_with_prior")
class MGH_Mammo_1Year_Risk_With_Prior(Abstract_MGH_Mammo_Cancer_With_Prior_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Risk_With_Prior, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Risk"

@RegisterDataset("mgh_mammo_1year_detection_with_prior")
class MGH_Mammo_1Year_Detection_With_Prior(Abstract_MGH_Mammo_Cancer_With_Prior_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(MGH_Mammo_1Year_Detection_With_Prior, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"
