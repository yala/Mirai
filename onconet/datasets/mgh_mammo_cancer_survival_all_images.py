import os
from collections import Counter
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.mgh_mammo_cancer_survival import MGH_Mammo_Cancer_Survival_Dataset, METADATA_FILENAMES
import onconet.utils
import tqdm
from random import shuffle
import numpy as np
import pdb

SUMMARY_MSG = "Contructed MGH Mammo {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
BIRADS_TO_PROB = {'NA':1.0, '1-Negative':0.0, '2-Benign':0.0, '0-Additional imaging needed':1.0, "3-Probably benign": 1.0, "4-Suspicious": 1.0, "5-Highly suspicious": 1.0, "6-Known malignancy": 1.0}
MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10

@RegisterDataset("mgh_mammo_risk_full_future_all_images")
class MGH_Mammo_Cancer_Survival_All_Images_Dataset(MGH_Mammo_Cancer_Survival_Dataset):
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
            # Sort exams by sdate
            exams = sorted(exams, key = lambda exam: exam['sdate'])

            for side in ['L', 'R']:
                all_images_so_far = []
                all_views_so_far = []
                all_sides_so_far = []
                time_stamps_so_far = []
                past_exams = []
                for i, exam in enumerate(exams):

                    prev_year = exams[i-1]['sdate'] if i != 0 else exams[0]['sdate']
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
                        if side == 'L':
                            right_ccs, right_mlos = [], []
                        elif side == 'R':
                            left_ccs, left_mlos = [], []
                        y, y_seq, y_mask, time_at_event = self.get_label(exam, side)

                        birads = int(BIRADS_TO_PROB[exam['birads']])

                        # Update time stamps
                        year_delta = year - prev_year
                        time_stamps_so_far = [t+year_delta for t in time_stamps_so_far]
                        # Update with new data
                        all_images_so_far.extend( right_ccs + right_mlos + left_ccs + left_mlos )
                        all_views_so_far.extend( [0]*len(right_ccs) + [1]*len(right_mlos) + [0]*len(left_ccs) + [1]*len(left_mlos) )
                        all_sides_so_far.extend( [0]*len(right_ccs) + [0]*len(right_mlos) + [1]*len(left_ccs) + [1]*len(left_mlos) )
                        time_stamps_so_far.extend( [0]*len(right_ccs + right_mlos + left_ccs + left_mlos) )
                        # Truncate with Max len exams
                        all_images_so_far = all_images_so_far[-self.args.num_images:]
                        all_views_so_far = all_views_so_far[-self.args.num_images:]
                        all_sides_so_far = all_sides_so_far[-self.args.num_images:]
                        time_stamps_so_far = time_stamps_so_far[-self.args.num_images:]

                        if time_at_event < 0: # Case of negative breast with 0 yr follow but pos other breast
                            continue

                        dist_key = 'neg'
                        if y_seq[0] == 1 and self.args.shift_class_bal_towards_imediate_cancers:
                            dist_key = 'pos:1'
                        elif y:
                            dist_key = 'pos:any'
                        if self.args.year_weighted_class_bal:
                            dist_key = "year={};{}".format(year, dist_key)

                        if len(all_images_so_far) < self.args.min_num_images:
                            continue

                        dataset.append({
                            'paths': pad_to_length(all_images_so_far, '<PAD>', self.args.num_images),
                            'y': y,
                            'y_mask': y_mask,
                            'y_seq': y_seq,
                            'time_at_event': time_at_event,
                            'birads': birads,
                            'year': year,
                            'additional': {},
                            'exam': exam['accession'],
                            'dist_key': dist_key,
                            'ssn': ssn,
                            'time_seq': pad_to_length(time_stamps_so_far, MAX_TIME , self.args.num_images),
                            'view_seq': pad_to_length(all_views_so_far, MAX_VIEWS , self.args.num_images),
                            'side_seq': pad_to_length(all_sides_so_far, MAX_SIDES, self.args.num_images)
                        })
        return dataset


@RegisterDataset("mgh_mammo_risk_full_future_all_images_both_sides")
class MGH_Mammo_Cancer_Survival_All_Images_Both_Sides_Dataset(MGH_Mammo_Cancer_Survival_Dataset):
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
            # Sort exams by sdate
            exams = sorted(exams, key = lambda exam: exam['sdate'])

            all_images_so_far = []
            all_views_so_far = []
            all_sides_so_far = []
            time_stamps_so_far = []
            past_exams = []
            for i, exam in enumerate(exams):

                prev_year = exams[i-1]['sdate'] if i != 0 else exams[0]['sdate']
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

                    if len(left_ccs+left_mlos+right_ccs+right_mlos) != 4:
                        continue

                    y, y_seq, y_mask, time_at_event = self.get_label(exam, 'Any')
                    y_l, y_seq_l, y_mask_l, time_at_event_l = self.get_label(exam, 'L')
                    y_r, y_seq_r, y_mask_r, time_at_event_r = self.get_label(exam, 'R')

                    birads = int(BIRADS_TO_PROB[exam['birads']])

                    # Update time stamps
                    year_delta = year - prev_year
                    time_stamps_so_far = [t+year_delta for t in time_stamps_so_far]
                    # Update with new data
                    all_images_so_far.extend( right_ccs + right_mlos + left_ccs + left_mlos )
                    all_views_so_far.extend( [0]*len(right_ccs) + [1]*len(right_mlos) + [0]*len(left_ccs) + [1]*len(left_mlos) )
                    all_sides_so_far.extend( [0]*len(right_ccs) + [0]*len(right_mlos) + [1]*len(left_ccs) + [1]*len(left_mlos) )
                    time_stamps_so_far.extend( [0]*len(right_ccs + right_mlos + left_ccs + left_mlos) )
                    # Truncate with Max len exams
                    all_images_so_far = all_images_so_far[-self.args.num_images:]
                    all_views_so_far = all_views_so_far[-self.args.num_images:]
                    all_sides_so_far = all_sides_so_far[-self.args.num_images:]
                    time_stamps_so_far = time_stamps_so_far[-self.args.num_images:]

                    if time_at_event < 0: # Case of negative breast with 0 yr follow but pos other breast
                        continue

                    dist_key = 'neg'
                    if y_seq[0] == 1 and self.args.shift_class_bal_towards_imediate_cancers:
                        dist_key = 'pos:1'
                    elif y:
                        dist_key = 'pos:any'
                    if self.args.year_weighted_class_bal:
                        dist_key = "year={};{}".format(year, dist_key)

                    if len(all_images_so_far) < self.args.min_num_images:
                        continue

                    dataset.append({
                        'paths': pad_to_length(all_images_so_far, '<PAD>', self.args.num_images),
                        'y': y,
                        'y_mask': y_mask,
                        'y_seq': y_seq,
                        'time_at_event': time_at_event,
                        'y_l': y_l,
                        'y_mask_l': y_mask_l,
                        'y_seq_l': y_seq_l,
                        'time_at_event_l': time_at_event_l,
                        'y_r': y_r,
                        'y_mask_r': y_mask_r,
                        'y_seq_r': y_seq_r,
                        'time_at_event_r': time_at_event_r,
                        'birads': birads,
                        'year': year,
                        'exam': exam['accession'],
                        'dist_key': dist_key,
                        'ssn': ssn,
                        'time_seq': pad_to_length(time_stamps_so_far, MAX_TIME , self.args.num_images),
                        'view_seq': pad_to_length(all_views_so_far, MAX_VIEWS , self.args.num_images),
                        'side_seq': pad_to_length(all_sides_so_far, MAX_SIDES, self.args.num_images),
                        'additionals': []
                    })
        return dataset

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)


