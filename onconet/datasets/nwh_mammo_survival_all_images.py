import os
from collections import Counter
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.nwh_mammo_survival import NWH_Mammo_Cancer_Survival
import onconet.utils
import tqdm
from random import shuffle
import numpy as np
import pdb

METADATA_FILENAMES = {
    'Risk': "/archive/nwh_metadata_feb20_2019.json",
    }
SUMMARY_MSG = "Contructed NWH Survival Mammo {} survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
BIRADS_TO_PROB = {'NA':1.0, '1-Negative':0.0, '2-Benign':0.0, '0-Additional imaging needed':1.0, "3-Probably benign": 1.0, "4-Suspicious": 1.0, "5-Highly suspicious": 1.0, "6-Known malignancy": 1.0}
MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10

@RegisterDataset("nwh_mammo_risk_full_future_all_images_both_sides")
class NWH_Mammo_Cancer_Survival_All_Images_Both_Sides_Dataset(NWH_Mammo_Cancer_Survival):
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
            ssn, images, split = mrn_row['mrn'], mrn_row['images'], mrn_row['split']
            if not split == split_group:
                continue

            all_images_so_far = []
            all_views_so_far = []
            all_sides_so_far = []
            time_stamps_so_far = []

            if not self.check_label(mrn_row):
                continue

            y, y_seq, y_mask, time_at_event = self.get_label(mrn_row)
            left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(images)
            if len(left_ccs+left_mlos) == 0 or len(right_ccs+right_mlos) == 0:
                continue
            # Update time stamps
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

            if len(all_images_so_far) < self.args.min_num_images:
                continue
            if not all( [img in self.path_to_hidden_dict for img in all_images_so_far]) and self.args.use_precomputed_hiddens:
                continue

            dataset.append({
                'paths': pad_to_length(all_images_so_far, '<PAD>', self.args.num_images),
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
                'birads': 0,
                'year': 0,
                'additionals': [],
                'exam': ssn,
                'dist_key': y,
                'ssn': ssn,
                'time_seq': pad_to_length(time_stamps_so_far, MAX_TIME , self.args.num_images),
                'view_seq': pad_to_length(all_views_so_far, MAX_VIEWS , self.args.num_images),
                'side_seq': pad_to_length(all_sides_so_far, MAX_SIDES, self.args.num_images)
            })
        return dataset

    def image_paths_by_views(self, images):
        left_ccs = [m['png_path'] for m in images if m['dicom_metadata']['SeriesDescription'] == 'L CC']
        right_ccs = [m['png_path'] for m in images if m['dicom_metadata']['SeriesDescription'] == 'R CC']
        left_mlos = [m['png_path'] for m in images if m['dicom_metadata']['SeriesDescription'] == 'L MLO']
        right_mlos = [m['png_path'] for m in images if m['dicom_metadata']['SeriesDescription'] == 'R MLO']
        return left_ccs, left_mlos, right_ccs, right_mlos

def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)


