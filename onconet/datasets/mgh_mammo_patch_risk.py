import os
from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.utils.region_annotation import parse_region_annotations, get_region_annotation_for_path
import onconet.utils
import tqdm
from random import shuffle
import pdb

METADATA_FILENAMES = {
    1 : "mammo_metadata_1year_apr6_2018.json",
}
SUMMARY_MSG = "Contructed {} {} dataset with {} records, and the following class balance \n {}"


@RegisterDataset("mgh_mammo_patch_detection")
class MGH_Mammo_Patch_Risk_Dataset(Abstract_Onco_Dataset):


    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        self.years = 1
        self.region_annotations = parse_region_annotations(args)
        super(MGH_Mammo_Patch_Risk_Dataset, self).__init__(args, transformer, split_group)

    @staticmethod
    def set_args(args):
        args.num_classes = 2


    def _add_sample(self, dataset, path, label, additional, class_balance, year, accession):
        class_balance.setdefault(label, 0)
        class_balance[label] += 1
        additional['label'] = label
        dataset.append({
            'path': path,
            'y': label,
            'additional': additional,
            'year': year,
            'exam': accession,
            'dist_key': "{}:{}".format(year, label)
        })

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        dataset = []
        class_balance = {}
        for mrn_row in tqdm.tqdm(self.metadata_json):
            split, exams = mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            for exam in exams:

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

                    label = self.get_label(exam)

                    for image_path in exam['files']:
                        bboxes = []
                        if label == 1:
                            region_annotation = get_region_annotation_for_path(image_path, self.region_annotations)

                            if not region_annotation['has_region_annotation']:
                                continue

                            min_x, max_x = region_annotation['region_bottom_left_x'], region_annotation['region_bottom_right_x']
                            width = max_x - min_x

                            center_x = (max_x + min_x) / 2
                            min_y, max_y = region_annotation['region_top_left_y'], region_annotation['region_bottom_left_y']
                            center_y = (min_y + max_y) / 2
                            height = max_y - min_y

                            bbox = {'boundary':
                                        {
                                          'min_x': min_x,
                                          'max_x': max_x,
                                          'center_x': center_x,
                                          'min_y': min_y,
                                          'max_y': max_y,
                                          'center_y':center_y,
                                          'width': width,
                                          'height': height
                                        }
                                    }

                            bboxes.append(bbox)

                            additional = {
                                            'height': 1,
                                            'width': 1,
                                            'boundary': bbox['boundary']
                                        }

                            self._add_sample(dataset, image_path, label, additional, class_balance, year, exam['accession'])

                        # Include 0 label for all images
                        label = 0
                        additional = {  'height': 1,
                                        'width': 1,
                                        'all_overlays': bboxes }

                        self._add_sample(dataset, image_path, label, additional, class_balance, year, exam['accession'])


        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(
            SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))
        return dataset

    @property
    def task(self):
        return "Patch Level {} Years Risk".format(self.years)

    def check_label(self, row):
        return 'years_to_cancer' in row

    def get_label(self, row):

        return row['years_to_cancer'] < self.years

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.years]


