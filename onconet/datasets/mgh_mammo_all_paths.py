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
import pickle

METADATA_FILENAMES = {
    'Risk': "mammo_metadata_all_years_only_breast_cancer_nov21_2019.json"
    }

SUMMARY_MSG = "Contructed MGH Mammo {} ALL_PATHs {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"

@RegisterDataset("mgh_mammo_all_paths_full_future")
class MGH_Mammo_All_Paths_Dataset(Abstract_Onco_Dataset):
    '''
        Working dataset for suvival analysis with all paths no exclusion.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []

        precomputed_hiddens_paths = pickle.load(open(self.args.hiddens_results_path,'rb'))['{}_hiddens'.format(split_group)][1] if self.args.use_precomputed_hiddens_in_get_hiddens else []
        precomputed_hiddens_paths = set(precomputed_hiddens_paths)

        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            for exam in exams:

                year = exam['sdate']

                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)
                left_label = self.get_label(exam, 'L')
                right_label = self.get_label(exam, 'R')

                for image_paths, label in [(left_ccs + left_mlos, left_label), (right_ccs + right_mlos, right_label)]:

                    y, y_seq, y_mask, time_at_event = label
                    if time_at_event < 0: # Case of negative breast with 0 yr follow but pos other breast
                        continue
                    for image_path in image_paths:

                        if image_path in precomputed_hiddens_paths:
                            continue

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

