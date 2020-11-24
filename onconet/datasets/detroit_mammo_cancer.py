import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
from onconet.learn.utils import BIRADS_TO_PROB
import pdb


METADATA_FILENAME = "/data/rsg/mammogram/detroit_data/Detroit/json_files/detroit_metadata_GE.json" #This path is located on the rosetta10 server

SUMMARY_MSG = "Contructed Henry Ford Mammo {} year {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
POSITIVE_PATIENT_AND_EXAM_MSG = "Number of positive exams: {}, number of positive patients: {}"


class Abstract_Detroit_Mammo_Cancer_Dataset(Abstract_Onco_Dataset):
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
                de_mrn = exam['DE_mrn']

                if self.check_label(exam):
                    for image_path in exam['png_paths']:
                        if self.args.mammogram_type == "GE":
                            if "GE_MEDICAL_SYSTEMS" not in image_path:
                                continue
                        elif self.args.mammogram_type == "Hologic":
                            if "GE_MEDICAL_SYSTEMS" in image_path:
                                continue

                        label = self.get_label(exam)
                        if label not in class_balance:
                            class_balance[label] = 0
                        class_balance[label] += 1
                        dataset.append({
                            'path': image_path,
                            'y': label,
                            'birads': 'n/a',
                            'additional': {},
                            'exam': exam['DE_acc'],
                            'dist_key': "{}".format(label),
                            'ssn': de_mrn
                        })


        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        exams = set([d['exam'] for d in dataset])
        positive_exams = set([d['exam'] for d in dataset if d['y']])
        patients = set([d['ssn'] for d in dataset])
        positive_patients = set([d['ssn'] for d in dataset if d['y']])
        positive_patient_paths = set([d['path'] for d in dataset if d['y']])
        print(
            SUMMARY_MSG.format(self.years, self.task, split_group, len(dataset), len(exams), len(patients), class_balance))
        print(POSITIVE_PATIENT_AND_EXAM_MSG.format(len(positive_exams), len(positive_patients)))
        return dataset

    def check_label(self, row):
        valid_pos = row['years_to_cancer'] < self.years
        valid_neg = row['years_to_follow_up'] >= self.years

        return (valid_pos or valid_neg)

    def get_label(self, row):
        any_cancer = row['years_to_cancer'] < self.years

        return any_cancer

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

    @staticmethod
    def set_args(args):
        args.num_classes = 2

@RegisterDataset("detroit_mammo_2year_detection")
class Detroit_Mammo_2Year_Detection(Abstract_Detroit_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 2
        super(Detroit_Mammo_2Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"

@RegisterDataset("detroit_mammo_1year_detection")
class Detroit_Mammo_1Year_Detection(Abstract_Detroit_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 1
        super(Detroit_Mammo_1Year_Detection, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Detection"
