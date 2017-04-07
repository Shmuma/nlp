import enum

import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model

import tensorflow as tf


WINDOW_SIZE = 40
L1_SIZE = 256
#L2_SIZE = 512


class OutTime(enum.Enum):
    NotSpecified = ''
    PartTime = 'part-time-job'
    FullTime = 'full-time-job'


class OutSalary(enum.Enum):
    NotSpecified = ''
    HourlyWage = 'hourle-wage'
    Salary = 'salary'


class OutEducation(enum.Enum):
    NotSpecified = ''
    AssocNeeded = 'associate-needed'
    BSNeeded = 'bs-degree-needed'
    MSNeeded = 'ms-or-phd-needed'
    LicenceNeeded = 'licence-needed'


class OutExperience(enum.Enum):
    NotSpecified = ''
    OneYear = '1-year-experience-needed'
    TwoYear = '2-4-years-experience-needed'
    FiveYear = '5-plus-years-experience-needed'


class OutSupervision(enum.Enum):
    NotSpecified = ''
    Supervision = 'supervising-job'


OUTPUT_ENUMS = (OutTime, OutSalary, OutEducation, OutExperience, OutSupervision)

OUTPUTS = {
    'time': OutTime,
    'salary': OutSalary,
    'edu': OutEducation,
    'exp': OutExperience,
    'supervise': OutSupervision
}


def create_model(batch_size, embeddings_len, output_kind):
    in_t = Input(batch_shape=(batch_size, WINDOW_SIZE, embeddings_len), name='input')

    out_rnn_t = LSTM(L1_SIZE, return_sequences=False, stateful=True, unroll=True, name='l1',
                     recurrent_dropout=0.5, dropout=0.7)(in_t)

    out_t = Dense(len(output_kind), name='out', activation='softmax')(out_rnn_t)
    return Model(inputs=in_t, outputs=out_t)


def pred_to_tags(pred, output_kind):
    """
    Convert predction array into tags list
    :param preds:
    :return: list of tag names
    """
    idx = np.argmax(pred)
    val = list(output_kind)[idx].value
    if val:
        return val
    return None


def tags_compare(pred_tag, true_tag, stp, sfp, sfn):
    """
    Perform tags vector comparison and update true positive, false positive and false negatives counters
    :param pred_tag:
    :param true_tag:
    :param stp:
    :param sfp:
    :param sfn:
    :return:
    """
    pred_l = np.argmax(pred_tag)
    true_l = np.argmax(true_tag)
    if pred_l == true_l:
        if pred_l != 0:
            stp += 1
    else:
        if pred_l != 0:
            sfp += 1
        if true_l != 0:
            sfn += 1
    return stp, sfp, sfn
