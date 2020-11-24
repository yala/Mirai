import os
from collections import Counter
import numpy as np
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.nwh_mammo_cancer import date_from_str
import onconet.utils
import tqdm
from random import shuffle
from onconet.learn.utils import BIRADS_TO_PROB
import pdb

METADATA_FILENAMES = {
    'Risk': "/archive/nwh_metadata_feb20_2019.json",
    }

SUMMARY_MSG = "Contructed NWH Survival Mammo {} survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"


@RegisterDataset("nwh_mammo_risk_full_future")
class NWH_Mammo_Cancer_Survival(Abstract_Onco_Dataset):
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
            y, y_seq, y_mask, time_at_event = label
            for image_path in paths:
                dataset.append({
                    'path': image_path,
                    'y': y,
                    'y_mask': y_mask,
                    'y_seq': y_seq,
                    'time_at_event': time_at_event,
                    'birads': 0,
                    'year': 0,
                    'additional': {},
                    'exam': ssn,
                    'dist_key': y,
                    'ssn': ssn
                })

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
        follow_up_date = date_from_str( row['last.appt.date'])
        cancer_date = inv_date if self.args.invasive_only else min(dcis_date, inv_date)

        follow_up = (follow_up_date - exam_date).days // 365
        cancer_years = (cancer_date - exam_date).days // 365

        y = cancer_years < self.args.max_followup
        y_seq = np.zeros(self.args.max_followup)
        if y:
            time_at_event = cancer_years
            y_seq[time_at_event:] = 1
            if not self.args.mask_like_slice and self.args.linear_interpolate_risk:
                year_hazard = 1.0 / (time_at_event + 1)
                y_seq = np.array([ (i+1)* year_hazard if v < 1.0 else v for i,v in enumerate(list(y_seq)) ])

        else:
            time_at_event = min(follow_up, self.args.max_followup) - 1

        if time_at_event == -1:
            pdb.set_trace()
        y_mask = np.array([1] * (time_at_event+1) + [0]* (self.args.max_followup - (time_at_event+1) ))
        if self.args.mask_like_slice and y:
            y_mask = np.zeros(self.args.max_followup)
            y_mask[time_at_event] = 1
        if (self.args.make_probs_indep or self.args.mask_like_indep) and y:
            y_mask =  np.ones(self.args.max_followup)
        assert len(y_mask) == self.args.max_followup

        return y, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event

    def get_summary_statement(self, dataset, split_group):
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        statement = SUMMARY_MSG.format(self.task, split_group, len(dataset), len(exams), len(patients), class_balance)
        statement += "\n" + "Censor Times: {}".format( Counter([d['time_at_event'] for d in dataset]))
        return statement

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Risk"

