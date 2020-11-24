import json
import numpy as np
import pdb
import copy
import torch
from scipy.special import binom

MISSING_VALUE = -1
HASNT_HAPPENED_VALUE = -5

RACE_CODE_TO_NAME = {
    1: 'White',
    2: 'African American',
    3: 'American Indian, Eskimo, Aleut',
    4: 'Asian or Pacific Islander',
    5: 'Other Race',
    6: 'Caribbean/West Indian',
    7: 'Unknown',
    8: 'Hispanic',
    9: 'Chinese',
    10: 'Japanese',
    11: 'Filipino',
    12: 'Hawaiian',
    13: 'Other Asian'
}
TREAT_MISSING_AS_NEGATIVE = False
NEGATIVE_99 = -99


class RiskFactorVectorizer():
    def __init__(self, args):

        self.risk_factor_metadata = parse_risk_factors(args)
        self.risk_factor_transformers = \
            {'binary_family_history': self.transform_binary_family_history,
             'binary_biopsy_benign': self.get_binary_occurence_transformer(
                 'biopsy_hyperplasia', 'biopsy_hyperplasia_age'),
             'binary_biopsy_LCIS': self.get_binary_occurence_transformer(
                 'biopsy_LCIS', 'biopsy_LCIS_age'),
             'binary_biopsy_atypical_hyperplasia': self.get_binary_occurence_transformer(
                 'biopsy_atypical_hyperplasia', 'biopsy_atypical_hyperplasia_age'),
             'age': self.get_exam_one_hot_risk_factor_transformer('age', [40, 50, 60, 70, 80]),
             'menarche_age': self.get_age_based_risk_factor_transformer('menarche_age', [10, 12, 14, 16]),
             'menopause_age': self.get_age_based_risk_factor_transformer('menopause_age', [45, 50, 55, 60]),
             'first_pregnancy_age': self.get_age_based_risk_factor_transformer( 'first_pregnancy_age', [20, 25, 30, 35, 40]),
             'density': self.get_image_biomarker_transformer('density'),
             'bpe': self.get_image_biomarker_transformer('bpe'),
             '5yearcancer': self.get_binary_transformer('5yearcancer'),
             'prior_hist': self.get_binary_transformer('prior_hist'),
             'years_to_cancer': self.get_exam_one_hot_risk_factor_transformer('years_to_cancer', [0, 1, 2, 3, 4, 10]),
             'race': self.transform_race,
             'parous': self.transform_parous,
             'menopausal_status': self.transform_menopausal_status,
             'weight': self.get_exam_one_hot_risk_factor_transformer('weight', [100, 130, 160, 190, 220, 250]),
             'height': self.get_exam_one_hot_risk_factor_transformer('height', [50, 55, 60, 65, 70, 75]),
             'ovarian_cancer': self.get_binary_occurence_transformer('ovarian_cancer',
                                                                     'ovarian_cancer_age'),
             'ovarian_cancer_age': self.get_age_based_risk_factor_transformer('ovarian_cancer_age',[30, 40, 50, 60, 70]),
             'ashkenazi': self.get_binary_transformer('ashkenazi', use_patient_factors=True),
             'brca': self.transform_brca,
             'mom_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('M'),
             'm_aunt_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('MA'),
             'p_aunt_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('PA'),
             'm_grandmother_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('MG'),
             'p_grantmother_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('PG'),
             'brother_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('B'),
             'father_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('F'),
             'daughter_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('D'),
             'sister_bc_cancer_history': self.get_binary_relative_cancer_history_transformer('S'),
             'mom_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('M', cancer='ovarian_cancer'),
             'm_aunt_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('MA', cancer='ovarian_cancer'),
             'p_aunt_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('PA', cancer='ovarian_cancer'),
             'm_grandmother_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('MG', cancer='ovarian_cancer'),
             'p_grantmother_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('PG', cancer='ovarian_cancer'),
             'sister_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('S', cancer='ovarian_cancer'),
             'daughter_oc_cancer_history': self.get_binary_relative_cancer_history_transformer('D', cancer='ovarian_cancer'),
             'hrt_type': self.get_hrt_information_transformer('type'),
             'hrt_duration': self.get_hrt_information_transformer('duration'),
             'hrt_years_ago_stopped': self.get_hrt_information_transformer('years_ago_stopped')
             }

        self.risk_factor_keys = args.risk_factor_keys
        self.feature_names = []
        self.risk_factor_key_to_num_class = {}
        for k in self.risk_factor_keys:
            if k not in self.risk_factor_transformers.keys():
                raise Exception("Risk factor key '{}' not supported.".format(k))
            names = self.risk_factor_transformers[k](None, None, just_return_feature_names=True)
            self.risk_factor_key_to_num_class[k] = len(names)
            self.feature_names.extend(names)
        args.risk_factor_key_to_num_class = self.risk_factor_key_to_num_class

    @property
    def vector_length(self):
        return len(self.feature_names)

    def get_feature_names(self):
        return copy.deepcopy(self.feature_names)

    def one_hot_vectorizor(self, value, cutoffs):
        one_hot_vector = torch.zeros(len(cutoffs) + 1)
        if value == MISSING_VALUE:
            return one_hot_vector
        for i, cutoff in enumerate(cutoffs):
            if value <= cutoff:
                one_hot_vector[i] = 1
                return one_hot_vector
        one_hot_vector[-1] = 1
        return one_hot_vector

    def one_hot_feature_names(self, risk_factor_name, cutoffs):
        feature_names = [""] * (len(cutoffs) + 1)
        feature_names[0] = "{}_lt_{}".format(risk_factor_name, cutoffs[0])
        feature_names[-1] = "{}_gt_{}".format(risk_factor_name, cutoffs[-1])
        for i in range(1, len(cutoffs)):
            feature_names[i] = "{}_{}_{}".format(risk_factor_name, cutoffs[i - 1], cutoffs[i])
        return feature_names

    def get_age_based_risk_factor_transformer(self, risk_factor_key, age_cutoffs):
        def transform_age_based_risk_factor(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, age_cutoffs)

            # if age-based risk factor, like menopause_age or first_pregnancy_age, is after the age at the exam, then treat it like it has not happened yet.
            exam_age = int(exam_factors['age'])
            age_based_risk_factor = int(patient_factors[risk_factor_key])
            if exam_age != MISSING_VALUE and exam_age < age_based_risk_factor:
                age_based_risk_factor = MISSING_VALUE  # effectively same as missing
            return self.one_hot_vectorizor(age_based_risk_factor, age_cutoffs)

        return transform_age_based_risk_factor

    def get_exam_one_hot_risk_factor_transformer(self, risk_factor_key, cutoffs):
        def transform_exam_one_hot_risk_factor(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, cutoffs)
            risk_factor = int(exam_factors[risk_factor_key])
            return self.one_hot_vectorizor(risk_factor, cutoffs)

        return transform_exam_one_hot_risk_factor

    def get_binary_occurence_transformer(self, occurence_key, occurence_age_key):
        def transform_binary_occurence(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return ['binary_{}'.format(occurence_key)]
            binary_occurence = torch.zeros(1)
            occurence = int(patient_factors[occurence_key])
            occurence_age = int(patient_factors[occurence_age_key])
            exam_age = int(exam_factors['age'])
            if occurence and (occurence_age == MISSING_VALUE or exam_age >= occurence_age):
                binary_occurence[0] = 1
            return binary_occurence

        return transform_binary_occurence

    def get_binary_transformer(self, risk_factor_key, use_patient_factors=False):
        def transform_binary(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return ['binary_{}'.format(risk_factor_key)]
            binary_risk_factor = torch.zeros(1)
            risk_factor = int(patient_factors[risk_factor_key]) if use_patient_factors else int(
                exam_factors[risk_factor_key])
            # If a binary risk factor is -1, we also want to treat it as negative (0)
            binary_risk_factor[0] = 1 if risk_factor == 1 else 0
            return binary_risk_factor

        return transform_binary

    def get_binary_relative_cancer_history_transformer(self, relative_code, cancer='breast_cancer'):
        def transform_binary_relative_cancer_history(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return ['{}_{}_hist'.format(relative_code, cancer)]
            binary_relative_cancer_history = torch.zeros(1)
            relative_list = patient_factors['relatives'][relative_code]
            for rel in relative_list:
                if rel[cancer] == 1:
                    binary_relative_cancer_history[0] = 1
            return binary_relative_cancer_history

        return transform_binary_relative_cancer_history

    def get_image_biomarker_transformer(self, name):
        def image_biomarker_transformer(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return (["{}_{}".format(name, i) for i in range(1,5)])
            image_biomarker_vector = torch.zeros(4)
            image_biomarker = int(exam_factors[name])
            if image_biomarker != MISSING_VALUE:
                image_biomarker_vector[image_biomarker - 1] = 1
            return image_biomarker_vector

        return image_biomarker_transformer

    def transform_binary_family_history(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return (['binary_family_history'])
        relatives_dict = patient_factors['relatives']
        binary_family_history = torch.zeros(1)
        for relative, relative_list in relatives_dict.items():
            if len(relative_list) > 0:
                binary_family_history[0] = 1
        return binary_family_history

    def transform_parous(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return (['parous'])
        binary_parous = torch.zeros(1)
        exam_age = int(exam_factors['age'])
        binary_parous[0] = 1 if patient_factors['num_births'] != MISSING_VALUE else 0
        if patient_factors['first_pregnancy_age'] != MISSING_VALUE:
            binary_parous[0] = 1 if patient_factors['first_pregnancy_age'] < exam_age else 0
        return binary_parous

    def transform_race(self, patient_factors, exam_factors, just_return_feature_names=False):
        values = range(1, 14)
        race_vector = torch.zeros(len(values))
        if just_return_feature_names:
            return [RACE_CODE_TO_NAME[i] for i in values]
        race = int(patient_factors['race'])
        race_vector[race - 1] = 1
        return race_vector

    def transform_menopausal_status(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return ['pre', 'peri', 'post', 'unknown']
        exam_age = int(exam_factors['age'])
        menopausal_status = 3  # unknown
        age_at_menopause = patient_factors['menopause_age'] \
            if patient_factors['menopause_age'] != MISSING_VALUE else NEGATIVE_99
        if age_at_menopause != NEGATIVE_99:
            if age_at_menopause < exam_age:
                menopausal_status = 2
            elif age_at_menopause == exam_age:
                menopausal_status = 1
            elif age_at_menopause > exam_age:
                menopausal_status = 0
        else:
            if TREAT_MISSING_AS_NEGATIVE:
                menopausal_status = 0
        menopausal_status_vector = torch.zeros(4)
        menopausal_status_vector[menopausal_status] = 1
        return menopausal_status_vector

    def transform_brca(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return ['never or unknown', 'negative result', 'brca1', 'brca2']
        genetic_testing_patient = 0
        brca1 = patient_factors['brca1']
        brca2 = patient_factors['brca2']
        if brca2 == 1:
            genetic_testing_patient = 3
        elif brca1 == 1:
            genetic_testing_patient = 2
        elif brca1 == 0:
            genetic_testing_patient = 1
        genetic_testing_vector = torch.zeros(4)
        genetic_testing_vector[genetic_testing_patient] = 1
        return genetic_testing_vector

    def get_hrt_information_transformer(self, piece):
        def transform_hrt_information(patient_factors, exam_factors, just_return_feature_names=False):
            year_cutoffs = [1,3,5,7]
            piece_to_feature_names = {'type': ['hrt_combined', 'hrt_estrogen', 'hrt_unknown'],
                                      'duration': self.one_hot_feature_names('hrt_duration', year_cutoffs),
                                      'years_ago_stopped': self.one_hot_feature_names('hrt_years_ago_stopped', year_cutoffs)}
            assert piece in piece_to_feature_names.keys()
            if just_return_feature_names:
                return piece_to_feature_names[piece]

            hrt_vector = torch.zeros(3)

            duration = MISSING_VALUE
            hrt_type = MISSING_VALUE
            hrt_years_ago_stopped = MISSING_VALUE
            first_age_key = None
            last_age_key = None
            duration_key = None
            current_age = int(exam_factors['age'])
            if patient_factors['combined_hrt']:
                hrt_type = 0
                first_age_key = 'combined_hrt_first_age'
                last_age_key = 'combined_hrt_last_age'
                duration_key = 'combined_hrt_duration'
            elif patient_factors['estrogen_hrt']:
                hrt_type = 1
                first_age_key = 'estrogen_hrt_first_age'
                last_age_key = 'estrogen_hrt_last_age'
                duration_key = 'estrogen_hrt_duration'
            elif patient_factors['unknown_hrt']:
                hrt_type = 2
                first_age_key = 'unknown_hrt_first_age'
                last_age_key = 'unknown_hrt_last_age'
                duration_key = 'unknown_hrt_duration'

            if first_age_key:
                first_age = patient_factors[first_age_key]
                last_age = patient_factors[last_age_key]
                extracted_duration = patient_factors[duration_key]

                if last_age >= current_age and current_age != MISSING_VALUE:
                    if first_age != MISSING_VALUE and first_age > current_age:
                        # future_user
                        hrt_type = MISSING_VALUE
                    elif extracted_duration != MISSING_VALUE and last_age - extracted_duration > current_age:
                        # future_user
                        hrt_type = MISSING_VALUE
                    else:
                        duration = current_age - first_age if current_age != MISSING_VALUE and first_age != MISSING_VALUE else extracted_duration
                elif last_age != MISSING_VALUE:
                    hrt_years_ago_stopped = current_age - last_age
                    if extracted_duration != MISSING_VALUE:
                        duration = extracted_duration
                    elif first_age != MISSING_VALUE and last_age != MISSING_VALUE:
                        duration = last_age - first_age
                        assert  duration >= 0
                else:
                    duration = extracted_duration if extracted_duration != MISSING_VALUE else MISSING_VALUE

            if hrt_type > MISSING_VALUE:
                hrt_vector[hrt_type] = 1

            piece_to_feature_names = {'type': hrt_vector,
                                      'duration': self.one_hot_vectorizor(duration, year_cutoffs),
                                      'years_ago_stopped': self.one_hot_vectorizor(hrt_years_ago_stopped, year_cutoffs)}
            return piece_to_feature_names[piece]
        return transform_hrt_information

    def transform_5yearcancer(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return (['5yearcancer'])
        binary_5yearcancer = torch.zeros(1)
        binary_5yearcancer[0] = int(exam_factors['5yearcancer'])
        return binary_5yearcancer

    def transform(self, patient_factors, exam_factors):
        risk_factor_vecs = [self.risk_factor_transformers[key](patient_factors, exam_factors) for key in
                            self.risk_factor_keys]
        return risk_factor_vecs

    def get_risk_factors_for_sample(self, sample):
        sample_patient_factors = self.risk_factor_metadata[sample['ssn']]
        sample_exam_factors = self.risk_factor_metadata[sample['ssn']]['accessions'][sample['exam']]
        risk_factor_vector = self.transform(sample_patient_factors, sample_exam_factors)
        return risk_factor_vector


    def get_buckets_for_sample(self, sample):
        sample_patient_factors = self.risk_factor_metadata[sample['ssn']]
        sample_exam_factors = self.risk_factor_metadata[sample['ssn']]['accessions'][sample['exam']]
        buckets = {}
        for key in self.risk_factor_keys:
            names = self.risk_factor_transformers[key](None, None, just_return_feature_names=True)
            vectorized = self.risk_factor_transformers[key](sample_patient_factors, sample_exam_factors)
            if sum(vectorized) == 0:
                buckets[key] = 'missing_or_negative'
            else:
                name_index = int(vectorized.dot(torch.arange(len(vectorized))))
                buckets[key] = names[name_index]
        return buckets

        return self.transform(sample_patient_factors, sample_exam_factors)


def parse_risk_factors(args):
    '''
        Parse the risk factors json file and return a dict mapping ssns to patient dictionaries. Each patient dictionary
        contains patient-level risk factors (e.g. race), as well as an 'accessions' key, that maps to a dictionary
        mapping accesion#s to dictionaries containing exam-level risk factors (e.g. age).
    '''
    try:
        metadata_json = json.load(open(args.metadata_path, 'r'))
    except Exception as e:
        raise Exception("Not found {} {}".format(args.metadata_path, e))


    try:
        risk_factor_metadata = json.load(open(args.risk_factor_metadata_path, 'r'))
    except Exception as e:
        raise Exception(
            "Metadata file {} could not be parsed! Exception: {}!".format(args.risk_factor_metadata_path, e))

    if '5yearcancer' in args.risk_factor_keys:
        for patient in metadata_json:
            ssn = patient['ssn']
            for exam in patient['accessions']:
                acc = exam['accession']
                label = 1 if exam['label'] == 'POS' else 0
                risk_factor_metadata[ssn]['accessions'][acc]['5yearcancer'] = label

    if 'prior_hist' in args.risk_factor_keys:
        for patient in metadata_json:
            if 'nwh' in args.dataset:
                ssn = patient['mrn']
                risk_factor_metadata[ssn]['accessions'][ssn]['prior_hist'] = 0
            else:
                ssn = patient['ssn']
                for exam in patient['accessions']:
                    acc = exam['accession']
                    risk_factor_metadata[ssn]['accessions'][acc]['prior_hist'] = exam['prior_hist']
    if 'years_to_cancer' in args.risk_factor_keys:
        for patient in metadata_json:
            ssn = patient['ssn']
            for exam in patient['accessions']:
                acc = exam['accession']
                risk_factor_metadata[ssn]['accessions'][acc]['years_to_cancer'] = exam['years_to_cancer']
    if 'bpe' in args.risk_factor_keys:
        for patient in metadata_json:
            ssn = patient['ssn']
            for exam in patient['accessions']:
                acc = exam['accession']
                risk_factor_metadata[ssn]['accessions'][acc]['bpe'] = exam['bpe'] if 'bpe' in exam else MISSING_VALUE

    return risk_factor_metadata
