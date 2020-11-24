import imageio
import json
import numpy as np
import os
import warnings
from torch.utils import data
from onconet.datasets.factory import RegisterDataset

MP4_LOADING_ERR = "Error loading {}.\n{}"

@RegisterDataset("kinetics")
class Kinetics(data.Dataset):
    """A pytorch Dataset for the Kinetics dataset."""

    def __init__(self, args, transformers, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            transformers(list): A list of transformer objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(Kinetics, self).__init__()
        args.metadata_path = os.path.join(args.metadata_dir,
                                          self.METADATA_FILENAME)

        self.args = args
        self.transformers = transformers
        self.split_group = split_group

        with open(args.metadata_path, 'r') as f:
            metadata = json.load(f)

        for row in metadata:
            row['path'] = os.path.join(args.img_dir, row['path'])

        self.dataset = [row for row in metadata if row['split_group'] == split_group]

        labels = [row['label'] for row in self.dataset]
        labels = sorted(np.unique(labels))
        self.label_map = {label: index for index, label in enumerate(labels)}

    @staticmethod
    def set_args(args):
        args.num_classes = 400
        args.multi_image = True
        args.num_images = 32
        args.video = True

    @property
    def METADATA_FILENAME(self):
        return 'metadata.json'

    @property
    def NUM_FRAMES(self):
        return 32

    @property
    def STRIDE(self):
        return 2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            # Load video
            video = imageio.get_reader(sample['path'], 'ffmpeg')

            # Determine start of clip randomly where possible
            if len(video) <= self.NUM_FRAMES * self.STRIDE:
                frame_start = 0
            else:
                frame_start = np.random.randint(len(video) - self.NUM_FRAMES * self.STRIDE)

            # Select frames in clip and loop around to start if necessary
            x = [video.get_data((frame_start + i) % len(video)) for i in range(0, self.NUM_FRAMES, self.STRIDE)]

            for transformer in self.transformers:
                x = transformer(x)

            item = {
                'x': x,
                'y': self.label_map[sample['label']]
            }

            return item
        except Exception as e:
            warnings.warn(MP4_LOADING_ERR.format(sample['path'], e))
