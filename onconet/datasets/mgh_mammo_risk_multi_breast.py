from tqdm import tqdm
from onconet.datasets.abstract_onco_dataset import Abstract_Onco_Dataset
from onconet.datasets.factory import RegisterDataset
from onconet.utils.generic import normalize_dictionary

SUMMARY_MSG = "Contructed MGH Mammo {} {} dataset with {} records, and the following class balance \n {}"
METADATA_FILENAMES = {
    1: "mammo_metadata_1year_jul23_2018_0years_post_pos_path.json",
    2: "mammo_metadata_2year_multi_breast_jun4_2018.json",
    3: "mammo_metadata_3year_multi_breast_jun4_2018.json"
}

class Abstract_MGH_Mammo_Risk_Multi_Breast_Dataset(Abstract_Onco_Dataset):
    """
        A risk dataset where each input consists of two images of the same view on different breasts: i.e one MLO from the right breast, and one MLO from the left
    """

    def create_dataset(self, split_group, img_dir):
        """Gets the dataset from the paths and labels in the json.

        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
            img_dir(str): Not used

        Returns:
            The dataset, which is a list of dictionaries with each dictionary
            containing paths to the relevant images, the label, and any other
            additional information.
        """

        dataset = []
        class_balance = {}
        for mrn_row in tqdm(self.metadata_json):
            split, exams = mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            for exam in exams:
                if not self.check_label(exam):
                    continue

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

                # Get label
                label = self.get_label(exam)
                if label not in class_balance:
                    class_balance[label] = 0

                left_ccs, left_mlos, right_ccs, right_mlos = self.image_paths_by_views(exam)

                # Create data input with one left CC and one right CC
                if len(left_ccs) > 0 and len(right_ccs) > 0:
                    dataset.append({
                        'paths': [left_ccs[0], right_ccs[0]],
                        'y': label,
                        'additionals': [],
                        'exam': exam['accession'],
                        'dist_key': "{}:{}".format(year, label)
                    })
                    class_balance[label] += 1

                # Create data input with one left MLO and one right MLO
                if len(left_mlos) > 0 and len(right_mlos) > 0:
                    dataset.append({
                        'paths': [left_mlos[0], right_mlos[0]],
                        'y': label,
                        'additionals': [],
                        'exam': exam['accession'],
                        'dist_key': "{}:{}".format(year, label)
                    })
                    class_balance[label] += 1

        class_balance = normalize_dictionary(class_balance)
        print(SUMMARY_MSG.format(self.task, split_group, len(dataset), class_balance))

        return dataset

    @property
    def task(self):
        return "{} Years Risk Multi Breast".format(self.years)

    def check_label(self, row):
        return 'years_to_cancer' in row

    def get_label(self, row):
        return row['label'] == 'POS'

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.years]

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 2

@RegisterDataset("mgh_mammo_1year_risk_multi_breast")
class MGH_Mammo_1Year_Risk_Multi_Breast(Abstract_MGH_Mammo_Risk_Multi_Breast_Dataset):
    """A Pytorch Dataset object for the 1 years risk, multi breast.

    Screenings are defined as positive if within up to 1 year from the exam, there was
    a malignant report on the same breast side (or on both sides).
    Screenings are defined as negative if they didn't have a malignant report a year from the screening
    and had a benign report after at least 1 year (to make sure that they are still in the system).
    """

    def __init__(self, args, transformer, split_group):
        """Initializes the Dataset object.

        Constructs a standard PyTorch Datset object, which
        can be fed into a DataLoader for batching.

        Arguments:
            args(Namespace): Config.
            transformer(Transformer): A Transformer object, takes in a PIL image,
                performs some transformations, and returns a Tensor.
            split_group(str): One of ['train'|'dev'|'test'].
        """

        self.years = 1
        super(MGH_Mammo_1Year_Risk_Multi_Breast, self).__init__(args, transformer, split_group)

@RegisterDataset("mgh_mammo_2year_risk_multi_breast_exlude_1yr")
class MGH_Mammo_2Year_Risk_Multi_Breast_Exclude_1Year(Abstract_MGH_Mammo_Risk_Multi_Breast_Dataset):
    """A Pytorch Dataset object for the 2 years risk, multi breast.

    Screenings are defined as positive if within up to 2 year (and after 1yr) from the exam, there was
    a malignant report on the same breast side (or on both sides).
    Screenings are defined as negative if they didn't have a malignant report a year from the screening
    and had a benign report after at least 2 year (to make sure that they are still in the system).
    """


    def __init__(self, args, transformer, split_group):
        """Initializes the Dataset object.

        Constructs a standard PyTorch Datset object, which
        can be fed into a DataLoader for batching.

        Arguments:
            args(Namespace): Config.
            transformer(Transformer): A Transformer object, takes in a PIL image,
                performs some transformations, and returns a Tensor.
            split_group(str): One of ['train'|'dev'|'test'].
        """

        self.years = 2
        super(MGH_Mammo_2Year_Risk_Multi_Breast_Exclude_1Year, self).__init__(args, transformer, split_group)


    def check_label(self, row):
        return 'years_to_cancer' in row and row['years_to_cancer'] >= (self.years - 1)


@RegisterDataset("mgh_mammo_2year_risk_multi_breast")
class MGH_Mammo_2Year_Risk_Multi_Breast(Abstract_MGH_Mammo_Risk_Multi_Breast_Dataset):
    """A Pytorch Dataset object for the 2 years risk, multi breast.

    Screenings are defined as positive if within up to 2 year from the exam, there was
    a malignant report on the same breast side (or on both sides).
    Screenings are defined as negative if they didn't have a malignant report a year from the screening
    and had a benign report after at least 2 year (to make sure that they are still in the system).
    """

    def __init__(self, args, transformer, split_group):
        """Initializes the Dataset object.

        Constructs a standard PyTorch Datset object, which
        can be fed into a DataLoader for batching.

        Arguments:
            args(Namespace): Config.
            transformer(Transformer): A Transformer object, takes in a PIL image,
                performs some transformations, and returns a Tensor.
            split_group(str): One of ['train'|'dev'|'test'].
        """

        self.years = 2
        super(MGH_Mammo_2Year_Risk_Multi_Breast, self).__init__(args, transformer, split_group)


@RegisterDataset("mgh_mammo_3year_risk_multi_breast")
class MGH_Mammo_3Year_Risk_Multi_Breast(Abstract_MGH_Mammo_Risk_Multi_Breast_Dataset):
    """A Pytorch Dataset object for the 3 years risk, multi breast.

    Screenings are defined as positive if within up to 3 year from the exam, there was
    a malignant report on the same breast side (or on both sides).
    Screenings are defined as negative if they didn't have a malignant report a year from the screening
    and had a benign report after at least 3 year (to make sure that they are still in the system).
    """

    def __init__(self, args, transformer, split_group):
        """Initializes the Dataset object.

        Constructs a standard PyTorch Datset object, which
        can be fed into a DataLoader for batching.

        Arguments:
            args(Namespace): Config.
            transformer(Transformer): A Transformer object, takes in a PIL image,
                performs some transformations, and returns a Tensor.
            split_group(str): One of ['train'|'dev'|'test'].
        """

        self.years = 3
        super(MGH_Mammo_3Year_Risk_Multi_Breast, self).__init__(args, transformer, split_group)
