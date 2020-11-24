import os
from collections import Counter
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.kth_mammo_cancer_survival import KTH_Mammo_Cancer_Survival_Dataset
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
MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10
DAYS_IN_YEAR = 365
MIN_YEAR = 2009
MAX_YEAR = 2014
LAST_FOLLOWUP_DATE = datetime.datetime(year=2015, month=12, day=31)


@RegisterDataset("kth_mammo_risk_full_future_all_images")
class KTH_Mammo_Cancer_Survival_All_Images_Dataset(KTH_Mammo_Cancer_Survival_Dataset):

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
            ssn, split, exams = mrn_row['patient_id'], mrn_row['split_group'], mrn_row['accessions']
            if not split == split_group:
                continue

            for date_str, exam in exams.items():

                date = datetime.datetime.strptime( date_str, '%Y%m%d')
                year = date.year
                if (split == 'train') and (year < MIN_YEAR or year > MAX_YEAR):
                    continue
                exam['accession'] = "{}_{}".format(ssn, date_str)

                exam['years_to_cancer'] = exam['days_to_cancer'] // 365
                exam['years_to_last_followup'] = (LAST_FOLLOWUP_DATE - date).days // 365

                if self.check_label(exam):
                    left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                    if len(left_ccs+left_mlos+right_ccs+right_mlos) != 4:
                        continue

                    y, y_seq, y_mask, time_at_event = self.get_label(exam)

                    # Update with new data
                    all_images = ( right_ccs + right_mlos + left_ccs + left_mlos )
                    all_views =( [0]*len(right_ccs) + [1]*len(right_mlos) + [0]*len(left_ccs) + [1]*len(left_mlos) )
                    all_sides =  [0]*len(right_ccs) + [0]*len(right_mlos) + [1]*len(left_ccs) + [1]*len(left_mlos)
                    time_stamps =  [0]*len(all_images)

                    dist_key = 'neg'
                    if y_seq[0] == 1 and self.args.shift_class_bal_towards_imediate_cancers:
                        dist_key = 'pos:1'
                    elif y:
                        dist_key = 'pos:any'
                    if self.args.year_weighted_class_bal:
                        dist_key = "year={};{}".format(year, dist_key)

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
                        'year': year,
                        'exam': exam['accession'],
                        'dist_key': dist_key,
                        'ssn': ssn,
                        'time_seq': pad_to_length(time_stamps, MAX_TIME , self.args.num_images),
                        'view_seq': pad_to_length(all_views, MAX_VIEWS , self.args.num_images),
                        'side_seq': pad_to_length(all_sides, MAX_SIDES, self.args.num_images),
                        'additionals': []
                    })
        return dataset

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)


