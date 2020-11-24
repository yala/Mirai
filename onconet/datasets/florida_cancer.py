import os
from collections import OrderedDict
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
import pdb
import random

random.seed(0)

METADATA_FILENAME = "florida_metadata.json"
SUMMARY_MSG = "Constructed Florida {} {} dataset with {} records, and the following class balance \n {}"
UNRECOGNIZED_BREAST_ERR = "Could not determine {} of img {}.\n"

LABEL_MAP = {'benigns':0, 'normals':0, 'cancers':1}


class Abstract_Florida_Cancer_Dataset(Abstract_Onco_Dataset):

    @property
    def task(self):
        return "Cancer"

    def check_label(self, exam):
        # Don't include the benign_without_callbacks like was done by
        # https://www.synapse.org/#!Synapse:syn9773040/wiki/426908
        return 'label' in exam and exam['label'] in LABEL_MAP

    def get_label(self, image):
        '''
         Return that image contains cancer if exam has cancer and img has
         Malig finding.
        '''
        overlays = image['overlays']
        if len(overlays) == 0 or LABEL_MAP[image['exam_label']] == 0:
            return 0

        # Cancer exam with finding, now determine if correct breast
        for overlay in overlays:
            if overlay['pathology'] == 'MALIGNANT':
                return 1

        return 0

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

@RegisterDataset("florida_cancer")
class Florida_Cancer(Abstract_Florida_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformers - The transformers that should be applied to images.
        params: split_group - ['train'|'dev'|'test'].

        Constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        Labels:
        0 - benign
        1 - malignant

        * The dataset will contain the image path, and the label
        '''
        super(Florida_Cancer, self).__init__(args, transformer,
                                                   split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        random.shuffle(self.metadata_json)
        dataset = []
        class_balance = {}
        for exam in tqdm.tqdm(self.metadata_json):

            def fits_criteria(exam):
                group_valid = 'split_group' in exam and exam['split_group'] == split_group
                label_valid = self.check_label(exam)
                return group_valid and label_valid

            if not fits_criteria(exam):
                continue

            for img in exam['imgs']:
                img['exam_label'] = exam['label']
                label = self.get_label(img)

                dataset.append({
                               'path': img['file_path'],
                               'y': label,
                               'additional': {}
                               })

                if not label in class_balance:
                    class_balance[label] = 0

                class_balance[label] += 1


        class_balance = onconet.utils.generic.normalize_dictionary(
            class_balance)
        print(SUMMARY_MSG.format(self.task, split_group,
                                 len(dataset), class_balance))
        return dataset


@RegisterDataset("florida_cancer_multibreast")
class Florida_Cancer_Multi_Breast(Abstract_Florida_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformers - The transformers that should be applied to images.
        params: split_group - ['train'|'dev'|'test'].

        Constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        Labels:
        0 - benign
        1 - malignant

        * The dataset will contain the image path, and the label
        '''
        super(Florida_Cancer_Multi_Breast, self).__init__(args, transformer,
                                                   split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 2

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        random.shuffle(self.metadata_json)
        dataset = []
        class_balance = {}
        for exam in tqdm.tqdm(self.metadata_json):

            def fits_criteria(exam):
                group_valid = 'split_group' in exam and exam['split_group'] == split_group
                label_valid = self.check_label(exam)
                has_all_views = len(exam['imgs']) == 4
                return group_valid and label_valid and has_all_views

            if not fits_criteria(exam):
                continue

            ccs = {}
            mlos = {}
            for img in exam['imgs']:
                img['exam_label'] = exam['label']
                img['label'] = self.get_label(img)
                img['additional'] = {}
                if 'CC' in img['file_path']:
                    view = ccs
                elif 'MLO' in img['file_path']:
                    view = mlos
                else:
                    raise Exception(UNRECOGNIZED_BREAST_ERR.format(
                                                        'breast side',
                                                        img['file_path']))

                if 'RIGHT' in img['file_path']:
                    view['RIGHT'] = img
                elif 'LEFT' in img['file_path']:
                    view['LEFT'] = img
                else:
                    raise Exception(UNRECOGNIZED_BREAST_ERR.format(
                                                        'breast view',
                                                        img['file_path']))

            for view in [ccs, mlos]:
                imgs = [view[side] for side in ['LEFT', 'RIGHT']]
                label = max([img['label'] for img in imgs])
                img_paths = [img['file_path'] for img in imgs]
                dataset.append({
                               'paths': img_paths,
                               'y': label,
                               'additionals': []
                               })

                if not label in class_balance:
                    class_balance[label] = 0

                class_balance[label] += 1


        class_balance = onconet.utils.generic.normalize_dictionary(
            class_balance)
        print(SUMMARY_MSG.format(self.task, split_group,
                                 len(dataset), class_balance))
        return dataset

@RegisterDataset("florida_cancer_multiview")
class Florida_Cancer_Multiview(Abstract_Florida_Cancer_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformers - The transformers that should be applied to images.
        params: split_group - ['train'|'dev'|'test'].

        Constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        Labels:
        0 - benign
        1 - malignant

        * The dataset will contain the image path, and the label
        '''
        super(Florida_Cancer_Multiview, self).__init__(args, transformer,
                                                   split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 2

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        random.shuffle(self.metadata_json)
        dataset = []
        class_balance = {}
        for exam in tqdm.tqdm(self.metadata_json):

            def fits_criteria(exam):
                group_valid = 'split_group' in exam and exam['split_group'] == split_group
                label_valid = self.check_label(exam)
                has_all_views = len(exam['imgs']) == 4
                return group_valid and label_valid and has_all_views

            if not fits_criteria(exam):
                continue

            left_breast = OrderedDict()
            right_breast = OrderedDict()
            for img in exam['imgs']:
                img['exam_label'] = exam['label']
                img['label'] = self.get_label(img)
                img['additional'] = {}
                if 'LEFT' in img['file_path']:
                    breast = left_breast
                elif 'RIGHT' in img['file_path']:
                    breast = right_breast
                else:
                    raise Exception(UNRECOGNIZED_BREAST_ERR.format(
                                                        'breast side',
                                                        img['file_path']))

                if 'CC' in img['file_path']:
                    breast['cc'] = img
                elif 'MLO' in img['file_path']:
                    breast['mlo'] = img
                else:
                    raise Exception(UNRECOGNIZED_BREAST_ERR.format(
                                                        'breast view',
                                                        img['file_path']))

            for side in [left_breast, right_breast]:
                imgs = [side[view] for view in side.keys()]
                side_label = max([view['label'] for view in imgs])
                img_paths = [view['file_path'] for view in imgs]
                dataset.append({
                               'paths': img_paths,
                               'y': side_label,
                               'additionals': []
                               })

                if not side_label in class_balance:
                    class_balance[side_label] = 0

                class_balance[side_label] += 1


        class_balance = onconet.utils.generic.normalize_dictionary(
            class_balance)
        print(SUMMARY_MSG.format(self.task, split_group,
                                 len(dataset), class_balance))
        return dataset
