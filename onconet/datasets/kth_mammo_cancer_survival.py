import os
from collections import Counter
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
import numpy as np
import datetime
import pdb

METADATA_FILENAMES = {
    'Risk': "kth_metadata_with_train_and_test.json"
    }

SUMMARY_MSG = "Contructed KTH Mammo {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
DAYS_IN_YEAR = 365
MIN_YEAR = 2009
MAX_YEAR = 2014
LAST_FOLLOWUP_DATE = datetime.datetime(year=2015, month=12, day=31)

@RegisterDataset("kth_mammo_risk_full_future")
class KTH_Mammo_Cancer_Survival_Dataset(Abstract_Onco_Dataset):
    '''
        Working dataset for suvival analysis.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []
        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['patient_id'], mrn_row['split_group'], mrn_row['accessions']
            if not split == split_group:
                continue

            for date_str, exam in exams.items():

                date = datetime.datetime.strptime( date_str, '%Y%m%d')
                year = date.year
                if split == 'train' and (year < MIN_YEAR or year > MAX_YEAR):
                    continue
                exam['accession'] = "{}_{}".format(ssn, date_str)

                exam['years_to_cancer'] = exam['days_to_cancer'] // 365
                exam['years_to_last_followup'] = (LAST_FOLLOWUP_DATE - date).days // 365
#                if year < MIN_YEAR or year > MAX_YEAR:
#                    continue

                if self.check_label(exam):
                    left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)
                    image_paths = left_ccs + left_mlos + right_ccs+ right_mlos
                    label = self.get_label(exam)



                    y, y_seq, y_mask, time_at_event = label
                    if time_at_event < 0: # Case of negative breast with 0 yr follow but pos other breast
                        continue

                    for image_path in image_paths:
                        dist_key = 'neg'
                        if y_seq[0] == 1 and self.args.shift_class_bal_towards_imediate_cancers:
                            dist_key = 'pos:1'
                        elif y:
                            dist_key = 'pos:any'

                        dataset.append({
                            'path': image_path,
                            'y': y,
                            'y_mask': y_mask,
                            'y_seq': y_seq,
                            'time_at_event': time_at_event,
                            'year': year,
                            'additional': {},
                            'exam': exam['accession'],
                            'dist_key': dist_key,
                            'ssn': ssn
                        })
                
                 
        return dataset

    def get_summary_statement(self, dataset, split_group):
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        statement = SUMMARY_MSG.format(self.task, split_group, len(dataset), len(exams), len(patients), class_balance)
        statement += "\n" + "Censor Times: {}".format( Counter([d['time_at_event'] for d in dataset]))
        return statement

    def check_label(self, row):
        valid_pos = row['years_to_cancer'] < self.args.max_followup
        valid_neg = row['years_to_last_followup'] > 0

        return (valid_pos or valid_neg)

    def get_label(self, row):
        any_cancer = row["years_to_cancer"] < self.args.max_followup
        cancer_key = "years_to_cancer"

        y =  any_cancer
        y_seq = np.zeros(self.args.max_followup)

        if y:
            time_at_event = row[cancer_key]
            y_seq[row[cancer_key]:] = 1
            if not self.args.mask_like_slice and self.args.linear_interpolate_risk:
                year_hazard = 1.0 / (time_at_event + 1)
                y_seq = np.array([ (i+1)* year_hazard if v < 1.0 else v for i,v in enumerate(list(y_seq)) ])

        else:
            time_at_event = min(row["years_to_last_followup"], self.args.max_followup) - 1

        y_mask = np.array([1] * (time_at_event+1) + [0]* (self.args.max_followup - (time_at_event+1) ))
        if self.args.mask_like_slice and y:
            y_mask = np.zeros(self.args.max_followup)
            y_mask[time_at_event] = 1
        if (self.args.make_probs_indep or self.args.mask_like_indep) and y:
            y_mask =  np.ones(self.args.max_followup)
        assert len(y_mask) == self.args.max_followup
        return any_cancer, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Risk"

