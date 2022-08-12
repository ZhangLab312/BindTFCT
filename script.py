import os

import torch
import pandas as pd
import random
import numpy as np
np.set_printoptions(threshold=np.inf)
import math

from torch.utils.data.dataset import Dataset
from collections import Counter
from Utils.Embedding import HighOrderEncoding, seq2vec


class word2vec_dataset(Dataset):

    def __init__(self, k_mer_nucleotide, word2idx, skip_window, mer):
        super(word2vec_dataset, self).__init__()
        self.corpus_code = [word2idx[n] for n in k_mer_nucleotide]
        self.corpus_code = torch.tensor(self.corpus_code)

        self.skip_window = skip_window
        self.index_list = list(range(0, 4 ** mer))

    def __len__(self):
        return len(self.corpus_code)

    def __getitem__(self, index):
        c_word = self.corpus_code[index]
        bg_word_pos_indices = list(range(index - self.skip_window, index)) + \
                              list(range(index + 1, index + 1 + self.skip_window))
        bg_word_pos_indices = [i % len(self.corpus_code) for i in bg_word_pos_indices]
        bg_word_pos = self.corpus_code[bg_word_pos_indices]

        bg_word_neg = torch.multinomial(torch.tensor(list(set(self.index_list).difference(set(bg_word_pos.numpy()))),
                                                     dtype=float), bg_word_pos.shape[0], replacement=True)

        return c_word, bg_word_pos, bg_word_neg

    @staticmethod
    def tokenizer(seq_file_path, mer):
        # row_sequence = pd.read_csv(seq_file_path, sep=' ', header=None)
        row_sequence = pd.read_csv(seq_file_path, header=None)
        seq_num = row_sequence.shape[0]
        seq_len = len(row_sequence.loc[0, 1])

        k_mer_nucleotide = list()
        for i in range(0, seq_num):
            seq = row_sequence.loc[i, 1]
            for loc in range(seq_len - mer + 1):
                k_mer_nucleotide.append(seq[loc: loc + mer])

        k_mer_nucleotide = word2vec_dataset.re_sampling(k_mer_nucleotide, mer)

        return k_mer_nucleotide

    @staticmethod
    def word2idx(k_mer):
        base_pair = ['A', 'C', 'T', 'G']
        mapper = list([''])

        for _ in range(k_mer):
            mapper_previous = mapper.copy()
            for nucleotide_pre in range(len(mapper_previous)):
                for nucleotide_now in base_pair:
                    mapper.append(mapper_previous[nucleotide_pre] + nucleotide_now)

            for _ in range(len(mapper_previous)):
                mapper.pop(0)

        index = range(0, 4 ** k_mer)
        word2idx = dict()
        for i in range(len(mapper)):
            word2idx[mapper[i]] = index[i]

        return word2idx

    @staticmethod
    def re_sampling(k_mer_nucleotide, mer):
        m = 10 ** (-mer)

        c = dict(Counter(k_mer_nucleotide))
        w_freq = {w: c / len(k_mer_nucleotide) for w, c in c.items()}
        w_del_p = {w: min((m / c), 1) for w, c in w_freq.items()}
        w_del = list()
        for w, c in w_del_p.items():
            p = random.random()
            if c > p:
                w_del.append(w)

        new_corpus = [w for w in k_mer_nucleotide if w not in w_del]

        return new_corpus


class SampleReader:
    """
        文件的命名规范如下：
            对于Sequence文件夹：Train_seq.csv或者Test_seq.csv
            对于Shape文件夹：Train_shapename.csv或者Test_shapename.csv
            对于Histone文件夹：Train_his.csv或者Test_his.csv

        各个一级文件夹中，有三个二级文件夹，即Sequence文件夹，Shape文件夹和Histone文件夹

    """
    """
        SampleReader一次可读取一个文件夹下的一些文件，具体策略如下：
            get_seq()函数可以读取Sequence文件夹中有关的文件
            get_shape()函数可以读取Shape文件夹中有关的文件
            get_histone()函数可以读取Histone文件夹中有关的文件

        注：对于Train和Test，不能同时读取
    """

    def __init__(self, file_name):
        """
            file_path:
                ATF2
        """
        self.seq2vec = seq2vec()

        self.seq_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\sequence\\'
        self.shape_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\shape\\'
        self.histone_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\HM_101\\'

    def get_seq(self, seq_dim, Test=False):

        if Test is False:
            row_seq = pd.read_csv(self.seq_path + 'Train.csv', sep=',', header=None)
        else:
            row_seq = pd.read_csv(self.seq_path + 'TesT.csv', sep=',', header=None)

        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])

        completed_seqs = np.empty(shape=(seq_num, seq_len, seq_dim))

        completed_labels = np.empty(shape=(seq_num, 1))
        completed_d_labels = np.empty(shape=(seq_num, 1))
        for i in range(seq_num):
            completed_seqs[i] = self.seq2vec.converT(sequence=row_seq.loc[i, 1],
                                                     embeddinG_PATH=self.seq_path + 'word2vec-16.npy')
            """
            completed_seqs[i] = HighOrderEncoding.covert(
                sequence=row_seq.loc[i, 1], order=2, mapper=HighOrderEncoding.build_mapper(order=2))
            """
            completed_labels[i] = row_seq.loc[i, 2]
            completed_d_labels[i] = row_seq.loc[i, 3]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels, completed_d_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv',
                                                header=None))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv',
                                                header=None))

        """
            seq_num = shape_series[0].shape[0]
            seq_len = shape_series[0].shape[1]
        """
        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(shape_samples.shape[0]):
                completed_shape[m][i] = shape_samples.loc[m]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape

    def get_histone(self, Test=False):

        if Test is False:
            histone = pd.read_csv(self.histone_path + 'Train' + '.csv', header=None, index_col=None)
        else:
            histone = pd.read_csv(self.histone_path + 'Test' + '.csv', header=None, index_col=None)

        histone = histone.iloc[:, :]
        histone = histone.fillna(0)
        num = histone.shape[0] // 8
        histone = histone.values
        histone = np.array(np.split(histone, num))
        """
            mask
        """
        # histone[:, 0, :] = 0

        return histone


class SSDataset_690(Dataset):

    def __init__(self, file_name, seq_dim, Test=False):
        shapes = ['HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels, self.completed_d_labels = \
            sample_reader.get_seq(seq_dim=seq_dim, Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)
        self.completed_histone = sample_reader.get_histone(Test=Test)

        sequence = np.around(np.mean(self.completed_seqs, axis=0), decimals=3)
        shape = np.around(np.mean(self.completed_shape, axis=0), decimals=3)
        H_mod = np.around(np.mean(self.completed_histone, axis=0), decimals=3)

        np.savetxt('sequence_i.csv', sequence, delimiter=',')
        np.savetxt('shape_i.csv', shape, delimiter=',')
        np.savetxt('H_mod_i.csv', H_mod, delimiter=',')

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_histone[item], \
               self.completed_labels[item], self.completed_d_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]


# domain_dataseT_class = domain_dataseT(TF='ATF3', seq_dim=16, cell_num=5)
SSDataset_690('CTCF', 16, True)