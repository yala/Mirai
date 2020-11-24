import os
from collections import Counter
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
import numpy as np
import pdb

METADATA_FILENAMES = {
    'Risk': "mammo_metadata_all_years_only_breast_cancer_aug04_2018_with_years_since_cancer.json"
    }

SUMMARY_MSG = "Contructed MGH Mammo {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
BIRADS_TO_PROB = {'NA':0.0, '1-Negative':0.0, '2-Benign':0.0, '0-Additional imaging needed':1.0, "3-Probably benign": 1.0, "4-Suspicious": 1.0, "5-Highly suspicious": 1.0, "6-Known malignancy": 1.0}

@RegisterDataset("mgh_mammo_risk_full_future")
class MGH_Mammo_Cancer_Survival_Dataset(Abstract_Onco_Dataset):
    '''
        Working dataset for suvival analysis. Note, does not support invasive cancer yet.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []

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
                        if left_label[0] and not right_label[0]:
                            if len(left_ccs) + len(left_mlos) > 0:
                                right_ccs = []
                                right_mlos = []
                        if right_label[0] and not left_label[0]:
                            if len(right_ccs) + len(right_mlos) > 0:
                                left_ccs = []
                                left_mlos = []

                    for image_paths, label in [(left_ccs + left_mlos, left_label), (right_ccs + right_mlos, right_label)]:
                        y, y_seq, y_mask, time_at_event = label
                        if time_at_event < 0: # Case of negative breast with 0 yr follow but pos other breast
                            continue
                        for image_path in image_paths:
                            dist_key = 'neg'
                            if y_seq[0] == 1 and self.args.shift_class_bal_towards_imediate_cancers:
                                dist_key = 'pos:1'
                            elif y:
                                dist_key = 'pos:any'

                            if self.args.year_weighted_class_bal:
                                dist_key = "year={};{}".format(year, dist_key)

                            dataset.append({
                                'path': image_path,
                                'y': y,
                                'y_mask': y_mask,
                                'y_seq': y_seq,
                                'time_at_event': time_at_event,
                                'birads': birads,
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
        valid_history = True if not self.task == "Risk" or self.args.use_permissive_cohort else row['years_since_cancer'] > 0
        no_cancer_now = True if not self.task == "Risk" or self.args.use_permissive_cohort else row['years_to_cancer'] > 0

        return (valid_pos or valid_neg) and valid_history and no_cancer_now

    def get_label(self, row, side='Any'):
        year = row['sdate']
        inv_cancer = row["years_to_invasive_cancer"] < self.args.max_followup
        if side == 'Any':
            any_cancer = row["years_to_cancer"] < self.args.max_followup
            cancer_key = "years_to_cancer"
        elif side == 'L':
            any_cancer = row["left_years_to_cancer"] < self.args.max_followup
            cancer_key = "left_years_to_cancer"
        else:
            assert(side == 'R')
            any_cancer = row["right_years_to_cancer"] < self.args.max_followup
            cancer_key = "right_years_to_cancer"
        y = any_cancer and inv_cancer if self.args.invasive_only else any_cancer
        y_seq = np.zeros(self.args.max_followup)

        if y:
            time_at_event = row[cancer_key]
            y_seq[row[cancer_key]:] = 1

            if self.args.mask_mechanism == 'linear':
                year_hazard = 1.0 / (time_at_event + 1)
                y_seq = np.array([ (i+1)* year_hazard if v < 1.0 else v for i,v in enumerate(list(y_seq)) ])

        else:
            time_at_event = min(row["years_to_last_followup"], self.args.max_followup) - 1

        y_mask = np.array([1] * (time_at_event+1) + [0]* (self.args.max_followup - (time_at_event+1) ))
        if self.args.mask_mechanism == 'slice' and y:
            y_mask = np.zeros(self.args.max_followup)
            y_mask[time_at_event] = 1
        if (self.args.make_probs_indep or self.args.mask_mechanism == 'indep') and y:
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

