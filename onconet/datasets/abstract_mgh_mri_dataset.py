from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
import numpy as np
from collections import Counter

SUMMARY_MSG = "Constructed MGH MRI {} {} dataset with {} records, and the following class balance \n {}"


class Abstract_MGH_MRI_Dataset(Abstract_Onco_Dataset):
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        dataset = []
        class_balance = {}

        # set numpy random seed; doing this here guarantees train/dev will not overlap
        if self.args.ten_fold_cross_val and self.args.ten_fold_test_index != -1:
            np.random.seed(self.args.ten_fold_cross_val_seed)
            if split_group == 'dev':
                split_group = 'test' # dev same as test so we can see test performance during training

        for mrn_row in tqdm.tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['ssn'], mrn_row['split'], mrn_row['accessions']

            if self.args.ten_fold_cross_val and self.args.ten_fold_test_index != -1:
                if not ssn in self.args.patient_to_partition_dict:
                    self.args.patient_to_partition_dict[ssn] = np.random.choice(10, p=[0.1]*10)
                partition = self.args.patient_to_partition_dict[ssn]
                if partition == self.args.ten_fold_test_index:
                    split = 'test'
                else:
                    split = 'train'

            if not split == split_group:
                continue

            for exam in exams:
                if self.check_label(exam):
                    label = self.get_label(exam)
                    image_path = exam['files'][0]
                    dataset.append({
                        'ssn': mrn_row['ssn'],
                        'path': image_path,
                        'y': label,
                        'additional': {},
                        'exam': exam['accession']
                    })

        class_balance = Counter([d['y'] for d in dataset])
        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return self.metadata_json
