import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
from onconet.learn.utils import BIRADS_TO_PROB
import datetime
import pdb

METADATA_FILENAMES = {
    'Risk': "/archive/nwh_metadata_feb20_2019.json",
    }

SUMMARY_MSG = "Contructed NWH Mammo {} year {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"


class Abstract_NWH_Mammo_Cancer_Dataset(Abstract_Onco_Dataset):
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []

        class_balance = {}
        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, paths, split = mrn_row['mrn'], mrn_row['paths'], mrn_row['split']
            if not split == split_group:
                continue

            if split_group == 'train':
                continue
            elif split_group == 'dev':
                continue
            else:
                assert split_group == 'test'

            if not self.check_label(mrn_row):
                continue

            label = self.get_label(mrn_row)

            for image_path in paths:
                if label not in class_balance:
                    class_balance[label] = 0
                class_balance[label] += 1
                dataset.append({
                    'path': image_path,
                    'y': label,
                    'birads': 0,
                    'year': 0,
                    'additional': {},
                    'exam': ssn,
                    'dist_key': label,
                    'ssn': ssn
                })

        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        print(
            SUMMARY_MSG.format(self.years, self.task, split_group, len(dataset), len(exams), len(patients), class_balance))
        self.args.years_risk = self.years
        return dataset

    def check_label(self, row):
        real_screen_date = date_from_str(row['images'][0]['dicom_metadata']['StudyDate'])
        screen_date_valid =  date_from_str(row['bl.date']) == real_screen_date
        
        has_followup = (date_from_str(row['last.appt.date']) - real_screen_date).days // 365  > 0
        has_cancer = (min(date_from_str(row['date.DCIS']), date_from_str(row['date.BC'])) - real_screen_date).days // 365 < self.args.max_followup
        return screen_date_valid and (has_followup or has_cancer)

    def get_label(self, row):
        exam_date = date_from_str(row['images'][0]['dicom_metadata']['StudyDate'])
        dcis_date = date_from_str(row['date.DCIS'])
        inv_date = date_from_str(row['date.BC'])
        cancer_date = inv_date if self.args.invasive_only else min(dcis_date, inv_date)
        return ((cancer_date - exam_date).days // 365) < self.years

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 2

def date_from_str(date_str):
    if date_str == 'NA':
        return datetime.datetime(9999,1,1,0,0)
    else:
        if len(date_str) == 8:
            format_str = '%Y%m%d'
        elif len(date_str) == 10:
            format_str = '%Y-%m-%d'
        else:
            raise Exception("Format for {} not recognized!".format(date_str))
    return datetime.datetime.strptime(date_str, format_str)


@RegisterDataset("nwh_mammo_5year_risk")
class NWH_Mammo_5Year_Risk(Abstract_NWH_Mammo_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        self.years = 5
        super(NWH_Mammo_5Year_Risk, self).__init__(args, transformer, split_group)
    @property
    def task(self):
        return "Risk"





