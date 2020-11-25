import torch
import json
import numpy as np
import pdb
import copy
import random

REGION_ANNOTATION_FILE = "/home/administrator/Mounts/Isilon/metadata/mammo_1year_cancer_region_annotations.json"
REGION_ANNOTATION_FILE_NOTFOUND_ERR = "Region annotation file {} could not be parsed! Exception: {}!"

IMAGE_RIGHT_ALIGNED_PATH = "/home/administrator/Mounts/Isilon/metadata/image_path_to_right_aligned_aug22_2018.json"
try:
    IMAGE_RIGHT_ALIGNED = json.load(open(IMAGE_RIGHT_ALIGNED_PATH,'r'))
except Exception as e:
    IMAGE_RIGHT_ALIGNED = {}

BLANK_REGION_ANNOTATION = {
    'image_indx': -1,
    'region_bottom_left_x' : -1.0,
    'region_bottom_left_y' : -1.0,
    'region_bottom_right_x': -1.0,
    'region_bottom_right_y': -1.0,
    'region_top_left_x' : -1.0,
    'region_top_left_y' : -1.0,
    'region_top_right_x': -1.0,
    'region_top_right_y' : -1.0,
    'has_region_annotation': False
}

def get_annotation_mask(x, batch, volatile, args):
    '''
        Get a mask the size of x
    '''
    epsilon = 1e-6
    num_dim = len(x.size())
    if num_dim == 4:
        B, C, H, W = x.size() #[B, C, (T), H, W]
        T = 1
        # Temporarily inflate in T dim to make rest of logic consistent with multi image case
        x = x.unsqueeze(2)
    else:
        assert num_dim == 5
        B, C, T, H, W = x.size()


    half_w_step = 1/(W*2)
    half_h_step = 1/(H*2)

    if args.h_arr is None:
        h_arr = torch.arange(start=0, end=1-epsilon, step=1/H)
        args.h_arr = h_arr
    else:
        h_arr = args.h_arr

    if args.w_arr is None:
        w_arr = torch.arange(start=0, end=1-epsilon, step=1/W)
        args.w_arr = w_arr
    else:
        w_arr = args.w_arr

    h_arr = h_arr.unsqueeze(0).expand([B, 1, T, W, H]).transpose(3,4)
    w_arr = w_arr.unsqueeze(0).expand([B, 1, T, H, W])

    _left_x = torch.min(batch['region_bottom_left_x'], batch['region_top_left_x']).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
    _right_x = torch.max(batch['region_bottom_right_x'], batch['region_top_right_x']).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
    left_x = torch.min(_left_x, _right_x)
    right_x = torch.max(_left_x, _right_x)
    _top_y = torch.min(batch['region_top_left_y'], batch['region_top_right_y']).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
    _bottom_y = torch.max(batch['region_bottom_left_y'], batch['region_bottom_right_y']).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
    top_y = torch.min(_top_y, _bottom_y)
    bottom_y = torch.max(_top_y, _bottom_y)

    if num_dim == 4:
        left_x = left_x.unsqueeze(-1)
        right_x = right_x.unsqueeze(-1)
        top_y = top_y.unsqueeze(-1)
        bottom_y = bottom_y.unsqueeze(-1)

    w_mask  = ((w_arr+half_w_step) >= left_x).long() & ((w_arr - half_w_step) <= right_x).long()
    h_mask  = ((h_arr+half_h_step) >= top_y).long() & ((h_arr - half_h_step) <= bottom_y).long()

    mask = (w_mask.long() & h_mask.long()).long()

    if num_dim == 4:
        # Collapse out T dim for non-multi-image
        mask = mask.squeeze(2)

    return mask





def parse_region_annotations(args):
    '''
        Parse the region_annotation json and
        return a dict of img path to bounding box.
    '''

    try:
        region_metadata = json.load(open(REGION_ANNOTATION_FILE, 'r'))
    except Exception as e:
        raise Exception(REGION_ANNOTATION_FILE_NOTFOUND_ERR.format(REGION_ANNOTATION_FILE, e))

    random.shuffle(region_metadata)
    end_indx = int(len(region_metadata) * args.fraction_region_annotation_to_use)

    path_to_bboxes = {}
    for sample in region_metadata[:end_indx]:
        filename = sample['filename']
        if filename not in path_to_bboxes:
            path_to_bboxes[filename] = []

        path_to_bboxes[filename].extend(sample['bboxes'])

    return path_to_bboxes

def get_region_annotation_for_sample(sample, region_annotations, args):
    if args.multi_image:
        return [get_region_annotation_for_path(path, region_annotations, args, indx) for indx, path in enumerate(sample['paths'])]
    else:
        return get_region_annotation_for_path(sample['path'], region_annotations, args)


def get_region_annotation_for_path(path, region_annotations, args, image_indx=-1):
    region_annotation = copy.deepcopy(BLANK_REGION_ANNOTATION)

    if path in region_annotations:
        annotations = region_annotations[path]
        if len(annotations) > 0:
            annotation = annotations[0]
            region_annotation = {
                'image_indx': image_indx,
                'region_bottom_left_x' : annotation['bottom_left']['x'],
                'region_bottom_left_y' : annotation['bottom_left']['y'],
                'region_bottom_right_x': annotation['bottom_right']['x'],
                'region_bottom_right_y': annotation['bottom_right']['y'],
                'region_top_left_x' : annotation['top_left']['x'],
                'region_top_left_y' : annotation['top_left']['y'],
                'region_top_right_x': annotation['top_right']['x'],
                'region_top_right_y' : annotation['top_right']['y'],
                'has_region_annotation': True
            }


            if 'align_to_left' in args.image_transformers and IMAGE_RIGHT_ALIGNED[path]:
                assert 'align_to_left' in args.test_image_transformers
                region_annotation = flip_region_coords_left_right({'region_annotation':region_annotation})['region_annotation']

    return region_annotation

def flip_region_coords_left_right(additional):
    if additional is not None and 'region_annotation' in additional:
        region = additional['region_annotation']
        # Check if region annotation is defined
        if region['has_region_annotation']:
            orig_region = copy.deepcopy(region)
            # Remap region coordinates given left/right flip
            region['region_bottom_left_x'] = 1 - orig_region['region_bottom_right_x']
            region['region_bottom_right_x'] = 1 - orig_region['region_bottom_left_x']
            region['region_top_right_x'] = 1 - orig_region['region_top_left_x']
            region['region_top_left_x'] = 1 - orig_region['region_top_right_x']


def flip_region_coords_top_bottom(additional):
    if additional is not None and 'region_annotation' in additional:
        region = additional['region_annotation']

        # Check if region annotation is defined
        if region['has_region_annotation']:
            orig_region = copy.deepcopy(region)
            # Remap region coordinates given top/bottom flip
            region['region_bottom_left_y'] = 1 - orig_region['region_top_left_y']
            region['region_top_left_y'] = 1 - orig_region['region_bottom_left_y']
            region['region_bottom_right_y'] = 1 - orig_region['region_top_right_y']
            region['region_top_right_y'] = 1 - orig_region['region_bottom_right_y']


def rotate_region_coords_angle(angle, additional):
    if additional is not None and 'region_annotation' in additional:
        region = additional['region_annotation']
        if not region['has_region_annotation']:
            return

        theta = np.radians(angle)


        point_keys = ['region_bottom_left_{}', 'region_bottom_right_{}', 'region_top_left_{}', 'region_top_right_{}' ]

        points = [ np.array([region[key.format('x')], region[key.format('y')]])
                         for key in point_keys]
        points = np.array(points)

        center = np.array([.5, .5])

        # Shift coords so center is origin
        points = points - center
        # Transform via rotation matrix
        rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        points = np.dot( points, rotation_matrix.T )
        # Shift coords so top left is origin
        points = points + center

        # given rotated box, calculate circumscribing box
        min_x = np.min(points[:,0])
        max_x = np.max(points[:,0])
        min_y = np.min(points[:,1])
        max_y = np.max(points[:,1])

        region['region_bottom_left_x'] = min_x
        region['region_bottom_right_x'] = max_x
        region['region_top_left_x'] = min_x
        region['region_top_right_x'] = max_x
        region['region_bottom_left_y'] = max_y
        region['region_bottom_right_y'] = max_y
        region['region_top_left_y'] = min_y
        region['region_top_right_y'] = min_y





def make_region_annotation_blank(additional):
    if additional is not None and 'region_annotation' in additional:
        additional['region_annotation'] = copy.deepcopy(BLANK_REGION_ANNOTATION)

