import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils
import tqdm
import pdb
import random

random.seed(0)

METADATA_FILENAME = "florida_metadata.json"
SUMMARY_MSG = "Constructed Florida {} {} dataset with {} records, and the following class balance \n {}"
OVERLAY_LABEL_ERR = 'Unknown overlay label {} {}'

EXAM_TYPES = ['benigns', 'cancers',
              'normals']  # don't include benign_without_callback
LESION_TYPES = ['CALCIFICATION', 'MASS']  # don't include 'other'
PATHOLOGY_TYPES = ['BENIGN', 'MALIGNANT'
                   ]  # don't include benign_without_callback and unproven


@RegisterDataset("florida_patch_abnormalities")
class Florida_Patch_Abnormalities(Abstract_Onco_Dataset):
    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformers - The transformers that should be applied to images.
        params: split_group - ['train'|'dev'|'test'].

        Constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        Labels:
        0 - benign (Should be cropped from the image out of the overlays boundaries)
        1 - calcification benign
        2 - mass benign
        3 - calcification malignant
        4 - mass malignant

        * The dataset will contain the image path, the label and additional data in a dictionary
        that will contain the original width and height of the image, the overlay of the lesion
        for labels 1-4 and a list of all image overlays for label 0.
        '''
        super(Florida_Patch_Abnormalities, self).__init__(args, transformer,
                                                   split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 5

    @property
    def task(self):
        return "Patch abnormalities"

    def check_exam_label(self, exam):
        # Don't include the benign_without_callbacks like was done by
        # https://www.synapse.org/#!Synapse:syn9773040/wiki/426908
        return 'label' in exam and exam['label'] in EXAM_TYPES

    def check_label(self, overlay):
        # Don't include the benign_without_callbacks and unproven
        lesion_valid = 'leison_type' in overlay and overlay['leison_type'] in LESION_TYPES
        pathology_valid = 'pathology' in overlay and overlay['pathology'] in PATHOLOGY_TYPES
        return lesion_valid and pathology_valid

    def get_label(self, overlay):
        if overlay['pathology'] == 'BENIGN':
            if overlay['leison_type'] == 'CALCIFICATION':
                return 1
            elif overlay['leison_type'] == 'MASS':
                return 2
            else:
                raise (Exception(
                    OVERLAY_LABEL_ERR.format(overlay['pathology'], overlay[
                        'leison_type'])))
        elif overlay['pathology'] == 'MALIGNANT':
            if overlay['leison_type'] == 'CALCIFICATION':
                return 3
            elif overlay['leison_type'] == 'MASS':
                return 4
            else:
                raise (Exception(
                    OVERLAY_LABEL_ERR.format(overlay['pathology'], overlay[
                        'leison_type'])))
        else:
            raise (Exception(
                OVERLAY_LABEL_ERR.format(overlay['pathology'], overlay[
                    'leison_type'])))

    def _add_sample(self, dataset, img, label, additional, class_balance):
        class_balance.setdefault(label, 0)
        class_balance[label] += 1
        additional['label'] = label
        dataset.append({
            'path': img['file_path'],
            'y': label,
            'additional': additional
        })

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
                label_valid = self.check_exam_label(exam)
                return group_valid and label_valid

            if not fits_criteria(exam):
                continue

            for img in exam['imgs']:
                img_data = {
                    'height': img['height'],
                    'width': img['width'],
                }
                all_overlays = []
                for overlay in img['overlays']:
                    # Include overlay even if not valid for the label 0 case.
                    all_overlays.append(overlay)
                    if not self.check_label(overlay):
                        continue

                    label = self.get_label(overlay)
                    additional = img_data.copy()
                    additional['boundary'] = overlay['boundary']

                    self._add_sample(dataset, img, label, additional,
                                     class_balance)

                # Include a 0 label for all images
                label = 0
                additional = img_data.copy()
                additional['all_overlays'] = all_overlays

                self._add_sample(dataset, img, label, additional, class_balance)

        class_balance = onconet.utils.generic.normalize_dictionary(
            class_balance)
        print(SUMMARY_MSG.format(self.task, split_group,
                                 len(dataset), class_balance))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME
