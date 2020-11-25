import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import csv
import traceback
from collections import Counter
from onconet.datasets.loader.image import image_loader
from onconet.utils.region_annotation import parse_region_annotations, get_region_annotation_for_sample
from onconet.utils.risk_factors import parse_risk_factors, RiskFactorVectorizer
from scipy.stats import entropy
import pdb

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"

DEVICE_TO_ID = {'Lorad Selenia': 1,
                'Hologic Selenia': 1,
                'Senograph DS ADS_43.10.1':2,
                'Selenia Dimensions': 0,
                'Selenia Dimensions C-View':3}

DATASET_ITEM_KEYS = ['ssn', 'exam','birads', 'y_seq', 'y_mask', 'time_at_event', 'device', 'device_is_known',
            'time_seq', 'view_seq', 'side_seq', 'y_l', 'y_r', 'y_seq_r', 'y_mask_r', 'time_at_event_r', 'y_seq_l', 'y_mask_l', 'time_at_event_l']

class Abstract_Onco_Dataset(data.Dataset):
    """
    Abstract Object for all Onco Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Abstract_Onco_Dataset, self).__init__()


        if args.metadata_dir is not None and args.metadata_path is None:
            args.metadata_path = os.path.join(args.metadata_dir,
                                          self.METADATA_FILENAME)

        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(args.cache_path,
                                                      transformers)

        try:
            if 'json' in args.metadata_path:
                self.metadata_json = json.load(open(args.metadata_path, 'r'))
            else:
                assert 'csv' in args.metadata_path
                _reader = csv.DictReader(open(args.metadata_path,'r'))
                self.metadata_json = [r for r in _reader]
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))


        self.path_to_hidden_dict = {}
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
        if split_group == 'train' and self.args.data_fraction < 1.0:
            self.dataset = np.random.choice(self.dataset, int(len(self.dataset)*self.args.data_fraction), replace=False)
        try:
            self.add_device_to_dataset()
            if "all" not in self.args.allowed_devices:
                self.dataset = [d for d in self.dataset if (d['device_name'] if isinstance(d['device_name'], str) else d['device_name'][0])  in self.args.allowed_devices]
        except:
            print("Could not add device information to dataset")
        for d in self.dataset:
            if 'exam' in d and 'year' in d:
                args.exam_to_year_dict[d['exam']] = d['year']
            if 'device_name' in d and 'exam' in d:
                args.exam_to_device_dict[d['exam']] = d['device_name']
        print(self.get_summary_statement(self.dataset, split_group))
        if args.use_region_annotation:
            self.region_annotations = parse_region_annotations(args)
        args.h_arr, args.w_arr = None, None
        self.risk_factor_vectorizer = None
        if self.args.use_risk_factors:
            self.risk_factor_vectorizer = RiskFactorVectorizer(args)
            self.add_risk_factors_to_dataset()

        if 'dist_key' in self.dataset[0] and (args.year_weighted_class_bal or args.shift_class_bal_towards_imediate_cancers or args.device_class_bal):
            dist_key = 'dist_key'
        else:
            dist_key = 'y'

        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        if args.year_weighted_class_bal or args.class_bal:
            print("Label weights are {}".format(label_weights))
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]




    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @abstractmethod
    def check_label(self, row):
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        '''
        pass

    def get_summary_statement(self, dataset, split_group):
        '''
        Return summary statement
        '''
        return ""

    @abstractmethod
    def create_dataset(self, split_group, img_dir):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        pass


    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.args.use_precomputed_hiddens:
            return self.get_vector_item(index)
        else:
            return self.get_image_item(index)

    def get_vector_item(self, index):
        try:
            sample = self.dataset[index]

            def get_hidden(path):
                zero_vec = np.zeros( self.args.precomputed_hidden_dim)
                return  self.path_to_hidden_dict[path] if path in self.path_to_hidden_dict and not self.args.zero_out_hiddens else zero_vec

            hiddens_for_paths = np.array([get_hidden(path) for path in sample['paths']])
            x = torch.Tensor(hiddens_for_paths)


            item = {'x': x,
                    'y': sample['y']
                    }

            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            if self.args.use_risk_factors:
                item['risk_factors'] = sample['risk_factors']

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample['paths'], traceback.print_exc()))

    def get_image_item(self, index):
        sample = self.dataset[index]

        ''' Region annotation for each image. Dict for single image,
            list of dict for multi-image
        '''
        if self.args.use_region_annotation:
            region_annotation = get_region_annotation_for_sample(sample, self.region_annotations, self.args)
        try:
            if self.args.multi_image:
                additionals = sample['additionals']
                ''' Add region annotation to existing additionals
                    so transformers can mutate them as need be
                    (i.e if image flips or rotates)
                '''
                if self.args.use_region_annotation:
                    for img_index, path in enumerate(sample['paths']):
                        if img_index == len(additionals):
                            additionals.append({'region_annotation': region_annotation[img_index]})
                        else:
                            additionals[img_index]['region_annotation'] = region_annotation[img_index]
                x = self.image_loader.get_images(sample['paths'], additionals)
            else:
                additional = {} if sample['additional'] is None else sample['additional']
                if self.args.use_region_annotation:
                    additional['region_annotation'] = region_annotation
                x = self.image_loader.get_image(sample['path'], additional)

            item = {
                'x': x,
                'path': "\t".join(sample['paths']) if self.args.multi_image else sample['path'],
                'y': sample['y']
            }

            if self.args.use_region_annotation:
                if self.args.multi_image:
                    for coord in region_annotation[0]:
                        annotation_list = (lambda coord=coord, region_annotation=region_annotation: [img_annotation[coord] for img_annotation in region_annotation])()
                        item[coord] = torch.Tensor(annotation_list)
                else:
                    for coord in region_annotation:
                        item[coord] = region_annotation[coord]

            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            if self.args.use_risk_factors:
                item['risk_factors'] = sample['risk_factors']

            return item

        except Exception:
            if self.args.multi_image:
                warnings.warn(LOAD_FAIL_MSG.format(sample['paths'], traceback.print_exc()))
            else:
                warnings.warn(LOAD_FAIL_MSG.format(sample['path'], traceback.print_exc()))

    def add_risk_factors_to_dataset(self):
        for sample in self.dataset:
            sample['risk_factors'] = self.risk_factor_vectorizer.get_risk_factors_for_sample(sample)

    def add_device_to_dataset(self):
        path_to_device, exam_to_device = self.build_path_to_device_map()
        for d in self.dataset:

            paths = [d['path']] if 'path' in d else d['paths']
            d['device_name'], d['device'], d['device_is_known'] = [], [], []

            for path in paths:
                device = path_to_device[path]
                device_id = DEVICE_TO_ID[device] if device in DEVICE_TO_ID else 0
                device_is_known = device in DEVICE_TO_ID

                d['device_name'].append(device.replace(' ', '_') if device is not None else "<UNK>")
                d['device'].append(device_id)
                d['device_is_known'].append(device_is_known)

            single_image = len(paths) == 1
            if single_image:
                d['device_name'] = d['device_name'][0]
                d['device'] = d['device'][0]
                d['device_is_known'] = d['device_is_known'][0]
            else:
                d['device_name'] = np.array(d['device_name'])
                d['device'] = np.array(d['device'])
                d['device_is_known'] = np.array(d['device_is_known'], dtype=int)

        device_dist = Counter([ d['device'] if single_image else d['device'][-1] for d in self.dataset])
        print("Device Dist: {}".format(device_dist))
        if self.split_group == 'train':
            device_count = list(device_dist.values())
            self.args.device_entropy = entropy(device_count)
            print("Device Entropy: {}".format(self.args.device_entropy))

    def build_path_to_device_map(self):
        path_to_device = {}
        exam_to_device = {}
        for mrn_row in json.load(open('/Mounts/Isilon/metadata/mammo_metadata_all_years_only_breast_cancer_nov21_2019.json','r')):
            for exam in mrn_row['accessions']:
                exam_id = exam['accession']
                for file, device, view in zip(exam['files'], exam['manufacturer_models'], exam['views']):
                    device_name = '{} {}'.format(device, 'C-View') if 'C-View' in view else device
                    path_to_device[file] = device_name
                    exam_to_device[exam_id] = device_name
        return path_to_device, exam_to_device


    def image_paths_by_views(self, exam):
        '''
        Determine images of left and right CCs and MLO.
        Args:
        exam - a dictionary with views and files sorted relatively.

        returns:
        4 lists of image paths of each view by this order: left_ccs, left_mlos, right_ccs, right_mlos. Force max 1 image per view.

        Note: Validation of cancer side is performed in the query scripts/from_db/cancer.py in OncoQueries
        '''
        source_dir = '/home/{}'.format(self.args.unix_username) if self.args.is_ccds_server else ''

        def get_view(view_name):
            image_paths_w_view = [(view, image_path) for view, image_path in zip(exam['views'], exam['files']) if view.startswith(view_name)]

            if self.args.use_c_view_if_available:
                filt_image_paths_w_view = [(view, image_path) for view, image_path in image_paths_w_view if 'C-View' in view]
                if len(filt_image_paths_w_view) > 0:
                    image_paths_w_view = filt_image_paths_w_view
            else:
                image_paths_w_view = [(view, image_path) for view, image_path in image_paths_w_view if 'C-View' not in view]

            image_paths_w_view = image_paths_w_view[:1]
            image_paths = (lambda image_paths, source_dir: [source_dir+path for _ , path in image_paths_w_view])(image_paths_w_view, source_dir)
            return image_paths


        left_ccs = get_view('L CC')
        left_mlos = get_view('L MLO')
        right_ccs = get_view('R CC')
        right_mlos = get_view('R MLO')
        return left_ccs, left_mlos, right_ccs, right_mlos
