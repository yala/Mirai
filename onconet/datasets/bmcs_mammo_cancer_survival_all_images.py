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

SUMMARY_MSG = "Contructed BMCS Mammo {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10
DAYS_IN_YEAR = 365

@RegisterDataset("bmcs_all_full_future")
class BMCS_Mammo_Cancer_Survival_All_Images_Dataset_All_Device(Abstract_Onco_Dataset):

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
        if split_group != 'test':
            return dataset
        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, exams = mrn_row['pid'],  mrn_row['exams']

            for exam in exams:
                date_str = exam['sdate']
                exam['accession'] = "{}_{}".format(ssn, date_str)
                if "years_to_cancer" not in exam:
                    exam['years_to_cancer'] = exam['year_to_cancer']
                if exam['years_to_cancer'] < 0:
                    continue
                exam['years_to_last_followup'] = 5

                if self.check_label(exam):
                    left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                    if len(left_ccs+left_mlos+right_ccs+right_mlos) != 4:
                        continue

                    if not self.is_valid_device(exam):
                        continue

                    y, y_seq, y_mask, time_at_event = self.get_label(exam)

                    # Update with new data
                    all_images = ( right_ccs + right_mlos + left_ccs + left_mlos )
                    all_views =( [0]*len(right_ccs) + [1]*len(right_mlos) + [0]*len(left_ccs) + [1]*len(left_mlos) )
                    all_sides =  [0]*len(right_ccs) + [0]*len(right_mlos) + [1]*len(left_ccs) + [1]*len(left_mlos)
                    time_stamps =  [0]*len(all_images)

                    dataset.append({
                        'paths': pad_to_length(all_images, '<PAD>', self.args.num_images),
                        'y': y,
                        'y_mask': y_mask,
                        'y_seq': y_seq,
                        'time_at_event': time_at_event,
                        'y_l': y,
                        'y_mask_l': y_mask,
                        'y_seq_l': y_seq,
                        'time_at_event_l': time_at_event,
                        'y_r': y,
                        'y_mask_r': y_mask,
                        'y_seq_r': y_seq,
                        'time_at_event_r': time_at_event,
                        'year': 2008,
                        'exam': exam['accession'],
                        'ssn': ssn,
                        'time_seq': pad_to_length(time_stamps, MAX_TIME , self.args.num_images),
                        'view_seq': pad_to_length(all_views, MAX_VIEWS , self.args.num_images),
                        'side_seq': pad_to_length(all_sides, MAX_SIDES, self.args.num_images),
                        'additionals': []
                    })
        return dataset

    def image_paths_by_views(self, exam):

        def get_view(view_name):
            image_paths_w_view = [(view, image_path) for view, image_path in zip(exam['views'], exam['files']) if view.startswith(view_name)]
            image_paths_w_view = image_paths_w_view[:1]
            image_paths = [image_paths_w_view[-1][-1]] if len(image_paths_w_view) > 0 else []
            return image_paths

        left_ccs = get_view('LCC')
        left_mlos = get_view('LMLO')
        right_ccs = get_view('RCC')
        right_mlos = get_view('RMLO')
        return left_ccs, left_mlos, right_ccs, right_mlos

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

    def is_valid_device(self, row):
        return all([not '2000D' in man for man in row['manufacturer_model']])

    def get_label(self, row):
        any_cancer = row["years_to_cancer"] < self.args.max_followup
        cancer_key = "years_to_cancer"

        y =  any_cancer
        y_seq = np.zeros(self.args.max_followup)

        if y:
            time_at_event = int(row[cancer_key])
            y_seq[time_at_event:] = 1

        else:
            time_at_event = int(min(row["years_to_last_followup"], self.args.max_followup) - 1)

        y_mask = np.array([1] * (time_at_event+1) + [0]* (self.args.max_followup - (time_at_event+1) ))
        assert len(y_mask) == self.args.max_followup
        return any_cancer, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event

    @property
    def METADATA_FILENAME(self):
        return 'BMCS_metadata_sigmoid_final.json'#'BMCS_metadata_no_R2_CAD_final.json'

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    @property
    def task(self):
        return "Risk"


def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)

@RegisterDataset("bmcs_ge_all_full_future")
class BMCS_Mammo_Cancer_Survival_All_Images_Dataset_GE_All_Device(BMCS_Mammo_Cancer_Survival_All_Images_Dataset_All_Device):
    def is_valid_device(self, row):
        return all([not '2000D' in man and not 'Selenia' in man for man in row['manufacturer_model']])

@RegisterDataset("bmcs_ge_essential_full_future")
class BMCS_Mammo_Cancer_Survival_All_Images_Dataset_GE_Essential_Device(BMCS_Mammo_Cancer_Survival_All_Images_Dataset_All_Device):
    def is_valid_device(self, row):
        return all(['Essential' in man for man in row['manufacturer_model']])

@RegisterDataset("bmcs_hologic_full_future")
class BMCS_Mammo_Cancer_Survival_All_Images_Dataset_Hologic_Device(BMCS_Mammo_Cancer_Survival_All_Images_Dataset_All_Device):
    def is_valid_device(self, row):
        return all(['Selenia' in man for man in row['manufacturer_model']])
