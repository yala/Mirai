import torchvision
import random
import numpy as np
from PIL import Image, ImageStat, ImageOps, ImageFile
import pdb
from onconet.transformers.factory import RegisterImageTransformer
from onconet.transformers.abstract import Abstract_transformer
from onconet.utils.region_annotation import flip_region_coords_left_right, flip_region_coords_top_bottom, rotate_region_coords_angle, make_region_annotation_blank
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASS_NOT_SUPPORT_REGION_WARNING = "{} does not support region annotations! Bounding box coordinates removed to prevent incorrect behavior"

class CordRescaler():
    def __init__(self, scaled_w, scaled_h):
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h

    def get_xy(self, original_w, original_h, x, y):
        '''
        Compute the new x,y coordinates in an image that was rescaled
        from original_w  X original_h
        to self.scaled_w X self.scaled_h
        '''
        related_x = float(x) / original_w
        related_y = float(y) / original_h
        new_x = int(related_x * self.scaled_w)
        new_y = int(related_y * self.scaled_h)
        return new_x, new_y


class Point:
    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord


class Rectangle:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def intersects(self, other):
        return not (self.top_right.x < other.bottom_left.x
                    or self.bottom_left.x > other.top_right.x
                    or self.top_right.y < other.bottom_left.y
                    or self.bottom_left.y > other.top_right.y)


def in_overlays(x1, y1, patch_width, patch_height, overlays, scaler,
                img_origin_size):
    for overlay in overlays:
        if in_overlay(x1, y1, patch_width, patch_height, overlay, scaler,
                      img_origin_size):
            return True

    return False


def in_overlay(x1, y1, patch_width, patch_height, overlay, scaler,
               img_origin_size):
    patch = Rectangle(
        Point(x1, y1 + patch_height), Point(x1 + patch_width, y1))
    x, y = scaler.get_xy(img_origin_size[0], img_origin_size[1], overlay['boundary']['min_x'],
                      overlay['boundary']['max_y'])
    bottom_left = Point(x, y)
    x, y = scaler.get_xy(img_origin_size[0], img_origin_size[1], overlay['boundary']['max_x'],
                      overlay['boundary']['min_y'])
    top_right = Point(x, y)
    overlay = Rectangle(bottom_left, top_right)
    return patch.intersects(overlay)


@RegisterImageTransformer("extract_patch")
class ExtractPatch(Abstract_transformer):
    '''
    Extract a patch based on the label and overlays that are in the
    additional data dict.
    The size of the patch will be w/h - based on kwargs.
    The patch location will be decided by the label:
    0 - randomly out of the all the overlay polygons
    else - center of overlay
    '''

    def __init__(self, args, kwargs):
        super(ExtractPatch, self).__init__()
        assert len(kwargs.keys()) == 1
        self.patch_width, self.patch_height = args.patch_size
        self.zoom = float(kwargs['z'])
        self.width, self.height = args.img_size
        self.cord_rescaler = CordRescaler(self.width, self.height)
        self.patch_scaler = torchvision.transforms.Resize( ( self.patch_height, self.patch_width))
        self.random_cropper = torchvision.transforms.RandomCrop( (self.patch_width, self.patch_height) )

    def __call__(self, img, additional=None):
        label = additional['label']
        zoom_factor = np.random.uniform(
                                low = 1 - self.zoom, high = 1 + self.zoom )
        patch_width =  int(self.patch_width * zoom_factor)
        patch_height = int(self.patch_height * zoom_factor)
        if label == 0:
            x1, y1 = (0, 0)
            patch_in_overlay = True
            while patch_in_overlay:
                x1 = random.randint(0, self.width - patch_width)
                y1 = random.randint(0, self.height - patch_height)
                patch_in_overlay = in_overlays(
                    x1, y1, patch_width, patch_height,
                    additional['all_overlays'], self.cord_rescaler,
                    (additional['width'], additional['height']))

            return self.patch_scaler( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )
        else:

            x1, y1 = self.cord_rescaler.get_xy(additional['width'],
                                               additional['height'],
                                               additional['boundary']['center_x'],
                                               additional['boundary']['center_y'])

            if zoom_factor > 1:
                return self.random_cropper( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )
            else:
                return self.patch_scaler( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )


@RegisterImageTransformer("scale_2d")
class Scale_2d(Abstract_transformer):
    '''
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)
    '''

    def __init__(self, args, kwargs):
        super(Scale_2d, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)
        self.transform = torchvision.transforms.Resize((height, width))

    def __call__(self, img, additional=None):
        return self.transform(img.convert('I'))

@RegisterImageTransformer("scale_2d_with_fixed_aspect_ratio")
class Scale_2d_With_Fixed_Aspect_Ratio(Abstract_transformer):
    '''
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)

        Does it in 3 steps:
        1) determine if left or right (similar to align left)
    '''

    def __init__(self, args, kwargs):
        super(Scale_2d_With_Fixed_Aspect_Ratio, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)

        self.aspect_ratio = float(height) / float(width)

        self.scale_transform = torchvision.transforms.Resize((height, width))

        # Create black image
        mask_r = Image.new('1', args.img_size)
        # Paint right side in white
        mask_r.paste(1, ((mask_r.size[0] *3 // 4), 0, mask_r.size[0],
                         mask_r.size[1]))
        mask_l = mask_r.transpose(Image.FLIP_LEFT_RIGHT)

        self.mask_r = mask_r
        self.mask_l = mask_l
        self.black = Image.new('I', args.img_size)

    def __call__(self, img, additional=None):
        width, height = img.size
        expected_width = int( height / self.aspect_ratio)
        if expected_width != width:
            assert width < expected_width
            pad_delta = expected_width - width
            left_pad = (pad_delta, 0, 0, 0)
            right_pad = (0, 0, pad_delta, 0)
            # Figure out if pad left or right.
            left = img.copy()
            left.paste(self.black, mask = self.mask_l)
            left_sum = np.array(left.getdata()).sum()
            right = img.copy()
            right.paste(self.black, mask = self.mask_r)
            right_sum = np.array(right.getdata()).sum()
            if right_sum > left_sum:
                pad = left_pad
            else:
                pad = right_pad
            img = ImageOps.expand(img, pad)
            new_aspect_ratio = float(img.size[1]) / img.size[0]
            assert new_aspect_ratio == self.aspect_ratio
        return self.scale_transform(img)


@RegisterImageTransformer("rand_hor_flip")
class Random_Horizontal_Flip(Abstract_transformer):
    '''
    torchvision.transforms.RandomHorizontalFlip wrapper
    '''

    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0

    def __call__(self, img, additional=None):
        if random.random() < 0.5:
            flip_region_coords_left_right(additional)
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


@RegisterImageTransformer("rand_ver_flip")
class Random_Vertical_Flip(Abstract_transformer):
    '''
    random vertical flip.
    '''

    def __init__(self, args, kwargs):
        super(Random_Vertical_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0

    def __call__(self, img, additional=None):
        if random.random() < 0.5:
            flip_region_coords_top_bottom(additional)
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


@RegisterImageTransformer("random_crop")
class Random_Crop(Abstract_transformer):
    '''
        torchvision.transforms.RandomCrop wrapper
        size of cropping will be decided by the 'h' and 'w' kwargs.
        'padding' kwarg is also available.
    '''

    def __init__(self, args, kwargs):
        super(Random_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2,3]
        size = (int(kwargs['h']), int(kwargs['w']))

        padding = int(kwargs['padding']) if 'padding' in kwargs else 0
        self.transform = torchvision.transforms.RandomCrop(size, padding)


    def __call__(self, img, additional=None):
        if self.args.use_region_annotation and additional is not None and 'region_annotation' in additional:
            warnings.warn(CLASS_NOT_SUPPORT_REGION_WARNING.format(self.__class__))
            make_region_annotation_blank(additional)

        return self.transform(img)

@RegisterImageTransformer("rotate_range")
class Rotate_Range(Abstract_transformer):
    '''
    Rotate image counter clockwise by random
    kwargs['min'] - kwargs['max'] degrees.

    Example: 'rotate/min=-20/max=20' will rotate by up to +/-20 deg
    '''

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 2
        self.max_angle = int(kwargs['max'])
        self.min_angle = int(kwargs['min'])

    def __call__(self, img, additional=None):
        angle = random.randint(self.min_angle, self.max_angle)
        rotate_region_coords_angle(angle, additional)
        return img.rotate(angle)


@RegisterImageTransformer("rotate_90")
class Rotate_90(Abstract_transformer):
    '''
    Rotate image by 0/90/180/270 degrees randomly.
    '''

    def __init__(self, args, kwargs):
        super(Rotate_90, self).__init__()
        assert len(kwargs.keys()) == 0
        self.args = args
        self.rotations = [
            0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270
        ]

        self.rotation_to_angle = {
            0:0, Image.ROTATE_90:90, Image.ROTATE_180:180, Image.ROTATE_270:270
        }

    def __call__(self, img, additional=None):
        rotation = np.random.choice(self.rotations)
        if rotation:
            angle = self.rotation_to_angle[rotation]
            rotate_region_coords_angle(angle, additional)
            return img.transpose(rotation)
        else:
            return img


@RegisterImageTransformer("align_to_left")
class Align_To_Left(Abstract_transformer):
    '''
    Aligns all images s.t. the breast will face left.
    Note: this should be applied after the scaling since the mask
    is the size of args.img_size.
    torchvision.transforms.RandomHorizontalFlip wrapper
    '''

    def __init__(self, args, kwargs):
        super(Align_To_Left, self).__init__()
        assert len(kwargs.keys()) == 0

        self.set_cachable(args.img_size)

        # Create black image
        mask_r = Image.new('1', args.img_size)
        # Paint right side in white
        mask_r.paste(1, ((mask_r.size[0] *3 // 4), 0, mask_r.size[0],
                         mask_r.size[1]))
        mask_l = mask_r.transpose(Image.FLIP_LEFT_RIGHT)

        self.mask_r = mask_r
        self.mask_l = mask_l
        self.black = Image.new('I', args.img_size)

    def __call__(self, img, additional=None):
        left = img.copy()
        left.paste(self.black, mask = self.mask_l)
        left_sum = np.array(left.getdata()).sum()
        right = img.copy()
        right.paste(self.black, mask = self.mask_r)
        right_sum = np.array(right.getdata()).sum()
        if right_sum > left_sum:
            flip_region_coords_left_right(additional)
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img

@RegisterImageTransformer("grayscale")
class Grayscale(Abstract_transformer):
    '''
    Given PIL image, converts it to grayscale
    with args.num_chan channels.
    '''

    def __init__(self, args, kwargs):
        super(Grayscale, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable(args.num_chan)

        self.transform = torchvision.transforms.Grayscale(args.num_chan)

    def __call__(self, img, additional=None):
        return self.transform(img)
