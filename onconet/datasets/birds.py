from onconet.datasets.factory import RegisterDataset
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
import onconet.utils

# The birds dataset is build on the CUB 200 2011 dataset. More information on this dataset can be found here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
# The script for generating the metadata file used for building this dataset can be found in OncoData/scripts/birds_metadata/birds_metadata.py. 


METADATA_FILENAME = "/home/administrator/Mounts/Isilon/metadata/birds.json"
SUMMARY_MSG = "Contructed birds {} {} dataset with {} records, and the following class balance \n {}"

class Abstract_CUB_200_2011_Dataset(Abstract_Onco_Dataset):

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: dataset object for the CUB_200_2011 (birds) dataset with specified split
        '''

        super(Abstract_CUB_200_2011_Dataset, self).__init__(args, transformer, split_group)
        self.args.num_classes = 200

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        dataset = []
        class_balance = {}

        split_image_ids = self.metadata_json['split_map'][split_group]
        images = self.metadata_json['images']

        # maps class_ids to human-readable species labels
        self.label_map = self.metadata_json['info']['class_id_map']

        for i in split_image_ids:
            image_info = images[str(i)]

            label = self.get_label(image_info)
            image_path = image_info['path']
            auxiliary = self.get_additional(image_info)


            if label not in class_balance:
                class_balance[label] = 0
            class_balance[label] += 1
            dataset.append({
                'path': image_path,
                'y': label,
                'additional': {},
                'auxiliary': auxiliary,
                'exam': i # TODO: consider replacing 'exam' with more generic 'id'
            })

        class_balance = onconet.utils.generic.normalize_dictionary(class_balance)
        print(SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

    def get_auxiliary(self, row):
        # this function returns dictionary of auxiliary information for each bird data point (eg. part localizations or bounding boxes)
        pass


@RegisterDataset("CUB_200_2011_basic")
class CUB_200_2011_Basic(Abstract_CUB_200_2011_Dataset):
    """
    Pytorch Dataset object for the 200-way classification task on the CUB 200 dataset.
    """

    def __init__(self, args, transformer, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(CUB_200_2011_Basic, self).__init__(args, transformer, split_group)

    @property
    def task(self):
        return "Classification"

    def check_label(self, row):
        return 'label' in row and row['label'] in range(1,201)

    def get_label(self, row):
        return row['label'] - 1

    def get_auxiliary(self, row):
        return {}

