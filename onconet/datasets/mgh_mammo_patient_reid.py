import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
from random import shuffle
from onconet.learn.utils import BIRADS_TO_PROB
import pdb
import numpy as np

METADATA_FILENAME = "mammo_metadata_all_years_only_breast_cancer_aug04_2018_with_years_since_cancer.json"

SUMMARY_MSG = "Contructed MGH Mammo {} REID {} dataset with {} records, {} patients.\n"

class Abstract_MGH_Mammo_ReId_Dataset(Abstract_Onco_Dataset):
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        all_patients = {}

        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue
            for exam in exams:
                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)
                if len(left_ccs) == 0 or len(left_mlos) == 0 or len(right_ccs) == 0 or len(right_mlos) == 0:
                    continue
                if not ssn in all_patients:
                    all_patients[ssn] = []
                all_patients[ssn].append( {
                    'exam': exam['accession'],
                    'ssn': ssn,
                    'l_cc': left_ccs,
                    'r_cc': right_ccs,
                    'l_mlo': left_mlos,
                    'r_mlo': right_mlos
                    })

        dataset = []

        for ssn in tqdm.tqdm(all_patients):
            exams = all_patients[ssn]
            dataset.extend(self.get_samples_from_patient(exams, all_patients))

        patients = set([d['ssn'] for d in dataset])
        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), len(patients)))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.num_images = 2
        args.multi_image = True



@RegisterDataset("mgh_mammo_reid_same_view")
class MGH_Mammo_ReId_Same_View_Dataset(Abstract_MGH_Mammo_ReId_Dataset):
    def __init__(self, args, transformer, split_group):
        super(MGH_Mammo_ReId_Same_View_Dataset, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "SameView"


    def sample_negative(self, ssn, view, all_patients):
        target_ssn = np.random.choice(list(all_patients.keys()), size=1)[0]
        target_exam = np.random.choice(all_patients[target_ssn], size=1)[0]
        return np.random.choice(target_exam[view], size=1)[0]

    def get_samples_from_patient(self, exams, all_patients):
        pos, neg = [], []

        for view in ['l_cc', 'l_mlo', 'r_mlo', 'r_cc']:
            all_paths = []
            for e in exams:
                all_paths.extend(e[view])
            if len(all_paths) >= 2:
                pos_paths = np.random.choice(all_paths, size=2, replace=False).tolist()
                pos.append({
                        'paths': pos_paths,
                        'y': True,
                        'ssn': exams[0]['ssn'],
                        'additionals': []
                    })

                neg_path = self.sample_negative(exams[0]['ssn'], view, all_patients)
                neg_paths = [ np.random.choice(pos_paths, size=1)[0], neg_path]
                np.random.shuffle(neg_paths)
                neg.append({
                        'paths': neg_paths,
                        'y': False,
                        'ssn': exams[0]['ssn'],
                        'additionals': []
                    })
        return pos+neg


@RegisterDataset("mgh_mammo_reid_same_breast")
class MGH_Mammo_ReId_Same_Breast_Dataset(Abstract_MGH_Mammo_ReId_Dataset):
    def __init__(self, args, transformer, split_group):
        super(MGH_Mammo_ReId_Same_Breast_Dataset, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "SameBreast"

    def sample_negative(self, ssn, breast, all_patients):
        target_ssn = np.random.choice(list(all_patients.keys()), size=1)[0]
        target_exam = np.random.choice(all_patients[target_ssn], size=1)[0]
        views =  ['l_cc', 'l_mlo'] if breast == 'l' else ['r_cc', 'r_mlo']
        return np.random.choice(target_exam[np.random.choice(views)], size=1)[0]

    def get_samples_from_patient(self, exams, all_patients):
        pos, neg = [], []

        for breast, views in [ ('l', ('l_cc', 'l_mlo')),
                ('r', ('r_mlo', 'r_cc'))]:
            all_paths = []
            for e in exams:
                for v in views:
                    all_paths.extend(e[v])

            if len(all_paths) >= 2:
                pos_paths = np.random.choice(all_paths, size=2, replace=False).tolist()
                pos.append({
                        'paths': pos_paths,
                        'y': True,
                        'ssn': exams[0]['ssn'],
                        'additionals': []
                    })

                neg_path = self.sample_negative(exams[0]['ssn'], breast, all_patients)
                neg_paths = [np.random.choice(pos_paths, size=1)[0], neg_path]
                np.random.shuffle(neg_paths)
                neg.append({
                        'paths': neg_paths,
                        'y': False,
                        'ssn': exams[0]['ssn'],
                        'additionals': []
                    })
        return pos+neg


@RegisterDataset("mgh_mammo_reid_any_view")
class MGH_Mammo_ReId_Any_View_Dataset(Abstract_MGH_Mammo_ReId_Dataset):
    def __init__(self, args, transformer, split_group):
        super(MGH_Mammo_ReId_Any_View_Dataset, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "AnyView"

    def sample_negative(self, ssn, all_patients):
        target_ssn = np.random.choice(list(all_patients.keys()), size=1)[0]
        target_exam = np.random.choice(all_patients[target_ssn], size=1)[0]
        views =  ['l_cc', 'l_mlo', 'r_cc', 'r_mlo']
        return np.random.choice(target_exam[np.random.choice(views)], size=1)[0]

    def get_samples_from_patient(self, exams, all_patients):
        pos, neg = [], []
        ssn = exams[0]['ssn']

        views = [ 'l_cc', 'l_mlo', 'r_mlo', 'r_cc']
        all_paths = []
        for e in exams:
            for v in views:
                all_paths.extend(e[v])

        if len(all_paths) >= 2:
            pos_paths = np.random.choice(all_paths, size=2, replace=False).tolist()
            pos.append({
                    'paths': pos_paths,
                    'y': True,
                    'ssn': ssn,
                    'additionals': []
                })

            neg_path = self.sample_negative(ssn, all_patients)
            neg_paths = [ np.random.choice(pos_paths, size=1)[0], neg_path]
            np.random.shuffle(neg_paths)
            neg.append({
                    'paths': neg_paths,
                    'y': False,
                    'ssn': exams[0]['ssn'],
                    'additionals': []
                })
        return pos+neg
