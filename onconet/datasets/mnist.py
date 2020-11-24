import torch
from torch.utils import data
from torchvision import datasets
from PIL import Image
import numpy as np
import skimage
from scipy.stats import multivariate_normal
from onconet.datasets.factory import RegisterDataset
from random import shuffle
import numpy as np
from PIL import Image
import warnings
warnings.simplefilter("ignore")

@RegisterDataset("mnist")
class MNIST_Dataset(data.Dataset):
    """A pytorch Dataset for the MNIST data."""

    def __init__(self, args, transformers, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            transformers(list): A list of transformer objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(MNIST_Dataset, self).__init__()

        self.args = args
        self.transformers = transformers
        self.split_group = split_group

        if self.split_group == 'train':
            self.dataset = datasets.MNIST('mnist',
                                          train=True,
                                          download=True)
        else:
            mnist_test = datasets.MNIST('mnist',
                                        train=False,
                                        download=True)
            if self.split_group == 'dev':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

    @staticmethod
    def set_args(args):
        args.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        for transformer in self.transformers:
            x = transformer(x, additional=None)

        if self.args.multi_image:
            x = x.unsqueeze(1)
            x = torch.cat( [x] * self.args.num_images, dim=1)

        item = {
            'x': x,
            'y': y
        }

        return item


@RegisterDataset("mnist_binary")
class MNIST_Binary_Dataset(MNIST_Dataset):
    """A pytorch Dataset for the MNIST data with two classes [0-4,5-9]."""

    def __init__(self, args, transformers, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            transformers(list): A list of transformer objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(MNIST_Binary_Dataset, self).__init__(args, transformers, split_group)

        self.args = args
        self.transformers = transformers
        self.split_group = split_group
        self.class_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}

    def __getitem__(self, index):
        item = super(MNIST_Binary_Dataset, self).__getitem__(index)
        item['y'] = self.class_mapping[item['y'].item()]

        return item

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    def __len__(self):
        return len(self.dataset)

@RegisterDataset("mnist_binary_full_future")
class MNIST_Binary_Full_Future_Dataset(MNIST_Dataset):
    """A pytorch Dataset for the MNIST data with two classes [0-4,5-9]."""

    def __init__(self, args, transformers, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            transformers(list): A list of transformer objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(MNIST_Binary_Full_Future_Dataset, self).__init__(args, transformers, split_group)

        self.args = args
        self.transformers = transformers
        self.split_group = split_group
        self.class_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}

    def __getitem__(self, index):
        item = super(MNIST_Binary_Full_Future_Dataset, self).__getitem__(index)
        item['y'] = self.class_mapping[item['y'].item()]
        item['y_seq'] = torch.ones(self.args.max_followup) if item['y'] else torch.zeros(self.args.max_followup) 
        item['y_mask'] = torch.ones( self.args.max_followup)
        item['time_at_event'] = self.args.max_followup - 1

        return item

    @staticmethod
    def set_args(args):
        args.num_classes = 2

    def __len__(self):
        return len(self.dataset)



@RegisterDataset("mnist_noise")
class MNIST_Noise(MNIST_Dataset):
    """A PyTorch Dataset for the MNIST data placed as small images on a large background with noise."""

    def __getitem__(self, index):
        x, y = self.dataset[index]

        # Create black background and paste MNIST digit on it
        h, w = self.args.background_size
        background = Image.new('L', (h, w))
        location = (np.random.randint(w - x.size[1]), np.random.randint(h - x.size[0]))
        background.paste(x, location)
        x = background

        # Add noise
        if self.args.noise:
            x = np.asarray(x)
            x = skimage.util.random_noise(x, var=self.args.noise_var)
            x = skimage.img_as_ubyte(x)
            x = Image.fromarray(x)

        for transformer in self.transformers:
            x = transformer(x, additional=None)

        item = {
            'x': x,
            'y': y
        }

        return item

@RegisterDataset("biased_mnist")
class MNIST_Biased_Dataset(data.Dataset):
    """
        A pytorch biased Dataset for the biased MNIST data.
        The train distribution contains a different color square for each
        color, and the dev, test distribution contain regular mnist

    """

    NUM_UNBAISED_TRAIN = 5000

    def __init__(self, args, transformers, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            transformers(list): A list of transformer objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(MNIST_Biased_Dataset, self).__init__()

        self.args = args
        self.transformers = transformers
        self.split_group = split_group

        '''
            We define self.dataset to be the distribution we draw batches
            from. This is biased for train and unbiased for dev/test.

            self.ref_dataset refers to unbiased samples, which can used in
            various regulazation stategies.
            self.ref_dataset doesn't share samples with self.dataset in train,
            but does in dev/test

        '''
        if self.split_group == 'train':

            mnist_train = datasets.MNIST('data',
                                          train=True,
                                          download=True)
            mnist_train = [self.force_rbg(mnist_train[i]) for i in range(len(mnist_train)) ]
            self.dataset = [self.bias(mnist_train[i]) for i in range(len(mnist_train) - self.NUM_UNBAISED_TRAIN)]
            self.ref_dataset = [mnist_train[i] for i in range(len(mnist_train) - self.NUM_UNBAISED_TRAIN, len(mnist_train))]

        else:
            mnist_test = datasets.MNIST('data',
                                        train=False,
                                        download=True)

            mnist_test = [self.force_rbg(mnist_test[i]) for i in range(len(mnist_test)) ]
            if self.split_group == 'dev':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
                self.ref_dataset = self.dataset
            elif self.split_group == 'test':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))]
                self.ref_dataset = self.dataset
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

        shuffle(self.dataset)
        shuffle(self.ref_dataset)



    def force_rbg(self, mnist_sample):
        x, y = mnist_sample
        x_arr = np.asarray(x)
        rbg_x = Image.fromarray( np.stack([x_arr]*3, axis=2))

        return rbg_x, y



    def bias(self, mnist_sample):
        '''
            Change background color of image to unique color for class
        '''
        x, y = mnist_sample

        y_to_rbg = {
            0:(255, 0 , 0) ,
            1:(0, 255, 0) ,
            2:(0, 0, 255) ,
            3:(133, 195, 255),
            4:(216, 8, 232) ,
            5:(216, 50, 100) ,
            6:(150, 8, 100) ,
            7:(255, 255, 0) ,
            8:(0, 255, 255) ,
            9:(255, 0, 255)
        }

        color = y_to_rbg[y]

        x_arr = np.asanyarray(x)
        sum_x = np.sum(x_arr, axis=2)
        background = sum_x == 0
        biased_x_arr = x_arr.copy()
        biased_x_arr[background] = color
        biased_x = Image.fromarray(biased_x_arr)
        return biased_x, y



    @staticmethod
    def set_args(args):
        args.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        target_x, target_y = self.ref_dataset[ index % len(self.ref_dataset)]

        for transformer in self.transformers:
            x = transformer(x, additional=None)
            target_x = transformer(target_x, additional=None)

        if self.args.multi_image:
            x = x.unsqueeze(1)
            x = torch.cat( [x] * self.args.num_images, dim=1)
            target_x = target_x.unsqueeze(1)
            target_x = torch.cat( [target_x] * self.args.num_images, dim=1)

        item = {
            'x': x,
            'y': y,
            'target_x': target_x,
            'target_y': target_y
        }

        return item

@RegisterDataset("mnist_location")
class MNIST_Location(MNIST_Noise):
    """A PyTorch Dataset for the MNIST data but with number replaced by blobs to determine if the network is simply memorizing the location of the number rather than actually reading the number."""

    def make_blob(self, radius):
        x, y = np.mgrid[0:radius, 0:radius]
        points = np.dstack((x, y))
        rv = multivariate_normal(mean=[(radius-1)/2, (radius-1)/2], cov=[[2*radius, 0], [0, 2*radius]])
        probs = rv.pdf(points)
        blob = 255 * probs / np.max(probs)
        blob = blob.astype('uint8')
        blob = Image.fromarray(blob)

        return blob

    def __init__(self, args, transformers, split_group):
        super(MNIST_Location, self).__init__(args, transformers, split_group)
        blob = self.make_blob(radius=28) # mnist is 28x28
        self.dataset = [(blob, y) for x, y in self.dataset]
