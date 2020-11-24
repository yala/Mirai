import datetime
import hashlib

INVALID_DATE_STR = "Date string not valid! Received {}, and got exception {}"
ISO_FORMAT = '%Y-%m-%dT%H:%M:%S'
def normalize_dictionary(dictionary):
    '''
    Normalizes counts in dictionary
    :dictionary: a python dict where each value is a count
    :returns: a python dict where each value is normalized to sum to 1
    '''
    num_samples = sum([dictionary[l] for l in dictionary])
    for label in dictionary:
        dictionary[label] = dictionary[label]*1. / num_samples
    return dictionary


def iso_str_to_datetime_obj(iso_string):
    '''
    Takes a string of format "YYYY-MM-DDTHH:MM:SS" and
    returns a corresponding datetime.datetime obj
    throws an exception if this can't be done.
    '''
    try:
        return datetime.datetime.strptime(iso_string, ISO_FORMAT)
    except Exception as e:
        raise Exception(INVALID_DATE_STR.format(iso_string, e))

def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()
