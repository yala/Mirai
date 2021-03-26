import os
from collections import Counter, defaultdict
import torch
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
import numpy as np
import datetime
import pdb

SUMMARY_MSG = "Contructed CSV Mammo {} Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
MAX_VIEWS = 2
MAX_SIDES = 2
MAX_TIME=10

@RegisterDataset("csv_mammo_risk_all_full_future")
class CSV_Mammo_Cancer_Survival_All_Images_Dataset(Abstract_Onco_Dataset):

    '''
        Working dataset for suvival analysis. Note, does not support invasive cancer yet.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """


        dict_dataset = defaultdict(dict)
        for _row in self.metadata_json:
            row = {k.encode('ascii', 'ignore').decode(): v.encode('ascii', 'ignore').decode() for k,v in _row.items()}
            patient_id, exam_id, split  = row['patient_id'], row['exam_id'], row['split_group']
            view = "{} {}".format(row['laterality'], row['view'])
            accession = "{}\t{}".format(patient_id, exam_id)
            file = row['file_path']

            dict_dataset[patient_id]['split'] = split
            dict_dataset[patient_id]['pid'] = patient_id
            if 'exams' not in dict_dataset[patient_id]:
                dict_dataset[patient_id]['exams'] = {}
            if accession not in dict_dataset[patient_id]['exams']:
                dict_dataset[patient_id]['exams'][accession] = {
                    'years_to_cancer': int(float(row['years_to_cancer'])),
                    'years_to_last_followup': int(float(row['years_to_last_followup'])),
                    'views': [],
                    'files': [],
                    'accession': accession
                }
            dict_dataset[patient_id]['exams'][accession]['views'].append(view)
            dict_dataset[patient_id]['exams'][accession]['files'].append(file)

        metadata = dict_dataset.values()
        dataset = []

        for mrn_row in tqdm.tqdm(metadata):
            ssn, exams = mrn_row['pid'],  mrn_row['exams']

            if mrn_row['split'] != split_group:
                continue

            for accession, exam in exams.items():

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

                    dataset.append({
                        'paths': pad_to_length(all_images, '<PAD>', self.args.num_images),
                        'y': y,
                        'y_mask': y_mask,
                        'y_seq': y_seq,
                        'time_at_event': time_at_event,
                        'exam': exam['accession'],
                        'ssn': ssn,
                        'time_seq': pad_to_length(time_stamps, MAX_TIME , self.args.num_images),
                        'view_seq': pad_to_length(all_views, MAX_VIEWS , self.args.num_images),
                        'side_seq': pad_to_length(all_sides, MAX_SIDES, self.args.num_images),
                        'additionals': [],
                        ### For back compatiblity with risk models that predict Left and right risk seperately
                        'year': -1,
                        'y_l': y,
                        'y_mask_l': y_mask,
                        'y_seq_l': y_seq,
                        'time_at_event_l': time_at_event,
                        'y_r': y,
                        'y_mask_r': y_mask,
                        'y_seq_r': y_seq,
                        'time_at_event_r': time_at_event
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
        valid_pos = row['years_to_cancer'] < self.args.max_followup and row['years_to_cancer'] >= 0
        valid_neg = row['years_to_last_followup'] > 0

        return (valid_pos or valid_neg)

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
        return self.args.metadata_path

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.max_followup = 5
        args.risk_factor_keys = ['density', 'binary_family_history', 'binary_biopsy_benign', 'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia', 'age', 'menarche_age', 'menopause_age', 'first_pregnancy_age', 'prior_hist', 'race', 'parous', 'menopausal_status', 'weight','height', 'ovarian_cancer', 'ovarian_cancer_age', 'ashkenazi', 'brca', 'mom_bc_cancer_history', 'm_aunt_bc_cancer_history', 'p_aunt_bc_cancer_history', 'm_grandmother_bc_cancer_history', 'p_grantmother_bc_cancer_history', 'sister_bc_cancer_history', 'mom_oc_cancer_history', 'm_aunt_oc_cancer_history', 'p_aunt_oc_cancer_history', 'm_grandmother_oc_cancer_history', 'p_grantmother_oc_cancer_history', 'sister_oc_cancer_history', 'hrt_type', 'hrt_duration', 'hrt_years_ago_stopped']
        args.metadata_dir = None
        args.pred_risk_factors = True
        args.use_pred_risk_factors_at_test = True
        args.survival_analysis_setup = True
        args.num_images = 4
        args.multi_image = True
        args.min_num_images =  4
        args.class_bal = True
        args.test_image_transformers =  ["scale_2d", "align_to_left"]
        args.test_tensor_transformers =  ["force_num_chan_2d", "normalize_2d"]
        args.image_transformers =  ["scale_2d", "align_to_left", "rand_ver_flip", "rotate_range/min=-20/max=20"]
        args.tensor_transformers =  ["force_num_chan_2d", "normalize_2d"]

    @property
    def task(self):
        return "Risk"


def pad_to_length(arr, pad_token, max_length):
    arr = arr[-max_length:]
    return  np.array( [pad_token]* (max_length - len(arr)) + arr)


@RegisterDataset("csv_mammo_risk_full_future")
class CSV_Mammo_Cancer_Survival_Dataset(CSV_Mammo_Cancer_Survival_All_Images_Dataset):

    '''
        Working dataset for suvival analysis. Note, does not support invasive cancer yet.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """


        dict_dataset = defaultdict(dict)
        for _row in self.metadata_json:
            row = {k.encode('ascii', 'ignore').decode(): v.encode('ascii', 'ignore').decode() for k,v in _row.items()}
            patient_id, exam_id, split  = row['patient_id'], row['exam_id'], row['split_group']
            view = "{} {}".format(row['laterality'], row['view'])
            accession = "{}\t{}".format(patient_id, exam_id)
            file = row['file_path']

            dict_dataset[patient_id]['split'] = split
            dict_dataset[patient_id]['pid'] = patient_id
            if 'exams' not in dict_dataset[patient_id]:
                dict_dataset[patient_id]['exams'] = {}
            if accession not in dict_dataset[patient_id]['exams']:
                dict_dataset[patient_id]['exams'][accession] = {
                    'years_to_cancer': int(float(row['years_to_cancer'])),
                    'years_to_last_followup': int(float(row['years_to_last_followup'])),
                    'views': [],
                    'files': [],
                    'accession': accession
                }
            dict_dataset[patient_id]['exams'][accession]['views'].append(view)
            dict_dataset[patient_id]['exams'][accession]['files'].append(file)

        metadata = dict_dataset.values()
        dataset = []

        for mrn_row in tqdm.tqdm(metadata):
            ssn, exams = mrn_row['pid'],  mrn_row['exams']

            if mrn_row['split'] != split_group:
                continue

            for accession, exam in exams.items():

                if self.check_label(exam):
                    left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)



                    y, y_seq, y_mask, time_at_event = self.get_label(exam)

                    for path in  left_ccs + left_mlos + right_ccs + right_mlos:
                        dataset.append({
                            'path': path,
                            'y': y,
                            'y_mask': y_mask,
                            'y_seq': y_seq,
                            'time_at_event': time_at_event,
                            'exam': exam['accession'],
                            'ssn': ssn,
                            'additional': {},
                            ### For back compatiblity with risk models that predict Left and right risk seperately
                            'year': -1,
                            'y_l': y,
                            'y_mask_l': y_mask,
                            'y_seq_l': y_seq,
                            'time_at_event_l': time_at_event,
                            'y_r': y,
                            'y_mask_r': y_mask,
                            'y_seq_r': y_seq,
                            'time_at_event_r': time_at_event
                        })
        return dataset

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.max_followup = 5
        args.risk_factor_keys = ['density', 'binary_family_history', 'binary_biopsy_benign', 'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia', 'age', 'menarche_age', 'menopause_age', 'first_pregnancy_age', 'prior_hist', 'race', 'parous', 'menopausal_status', 'weight','height', 'ovarian_cancer', 'ovarian_cancer_age', 'ashkenazi', 'brca', 'mom_bc_cancer_history', 'm_aunt_bc_cancer_history', 'p_aunt_bc_cancer_history', 'm_grandmother_bc_cancer_history', 'p_grantmother_bc_cancer_history', 'sister_bc_cancer_history', 'mom_oc_cancer_history', 'm_aunt_oc_cancer_history', 'p_aunt_oc_cancer_history', 'm_grandmother_oc_cancer_history', 'p_grantmother_oc_cancer_history', 'sister_oc_cancer_history', 'hrt_type', 'hrt_duration', 'hrt_years_ago_stopped']
        args.metadata_dir = None
        args.pred_risk_factors = True
        args.use_pred_risk_factors_at_test = True
        args.survival_analysis_setup = True
        args.class_bal = True
        args.test_image_transformers =  ["scale_2d", "align_to_left"]
        args.test_tensor_transformers =  ["force_num_chan_2d", "normalize_2d"]
        args.image_transformers =  ["scale_2d", "align_to_left", "rand_ver_flip", "rotate_range/min=-20/max=20"]
        args.tensor_transformers =  ["force_num_chan_2d", "normalize_2d"]


