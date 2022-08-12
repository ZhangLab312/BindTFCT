import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import numpy as np
import torch.nn.functional

from tqdm import tqdm
from Samples.script import word2vec_dataset
import pandas as pd


# from torch.utils.tensorboard import SummaryWriter


class skip_gram(nn.Module):

    def __init__(self, mer_num, d_model):
        super(skip_gram, self).__init__()
        self.vocab_size = 4 ** mer_num
        self.d_model = d_model
        self.i_Embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.o_Embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, c_word, bg_word_pos, bg_word_neg):
        c_word_code = self.i_Embedding(c_word).unsqueeze(2)
        bg_word_pos_code = self.o_Embedding(bg_word_pos)
        bg_word_neg_code = self.o_Embedding(bg_word_neg)

        o_pos = torch.bmm(bg_word_pos_code, c_word_code).squeeze(2)
        o_neg = torch.bmm(bg_word_neg_code, -c_word_code).squeeze(2)

        loss_pos = nn.functional.logsigmoid(o_pos).sum(1)
        loss_neg = nn.functional.logsigmoid(o_neg).sum(1)
        loss = loss_pos + loss_neg

        return -loss

    def get_c_word_code(self):
        return self.i_Embedding.weight.data.cpu().numpy()


class Train_SkipGram:

    def __init__(self, mer_num, d_model):
        super(Train_SkipGram, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = skip_gram(mer_num=mer_num, d_model=d_model).to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=0.2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=2,
                                                              factor=0.1, verbose=True)

        """
            cd os.path.abspath(os.curdir)
            tensorboard --logdir "./"
        """
        # self.log_info = SummaryWriter(os.path.abspath(os.curdir)+'/log_info')

        self.batch_size = 128
        self.epochs = 5
        self.d_model = d_model

    def learn(self, samples_file_path_l, cell, tf):
        path = os.path.abspath(os.curdir)

        completed_k_mer_nucleotide = list()
        for samples_file_path in samples_file_path_l:
            k_mer_nucleotide = word2vec_dataset.tokenizer(seq_file_path=samples_file_path, mer=3)
            completed_k_mer_nucleotide.extend(k_mer_nucleotide)

        Train_Set = word2vec_dataset(completed_k_mer_nucleotide,
                                     word2vec_dataset.word2idx(k_mer=3), skip_window=2, mer=3)

        Train_Loader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                         batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(self.epochs):
            ProgressBar = tqdm(Train_Loader)

            loss_for_lr_decay = []
            for step, data in enumerate(ProgressBar):
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                c_indices, bg_pos_indices, bg_neg_indices = data

                loss = self.model(c_indices.to(self.device), bg_pos_indices.to(self.device),
                                  bg_neg_indices.to(self.device)).mean()
                ProgressBar.set_postfix(loss=loss)

                loss_for_lr_decay.append(loss)
                # self.log_info.add_scalar('Train_loss', scalar_value=loss,
                #                          global_step=step + epoch * len(ProgressBar))

                loss.backward()
                self.optimizer.step()

            self.scheduler.step(torch.mean(torch.Tensor(loss_for_lr_decay)))
        model_save_path = "F:\\SSHSite\\Word2VecSavedModels"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(self.model.state_dict(), "{}\\{}_word2vec.pth".format(model_save_path, tf))
        save_path = "{}\\{}\\sequence\\word2vec-{}".format(ROOT_PATH, tf, self.d_model)
        np.save(save_path, self.model.get_c_word_code())

        # self.log_info.close()


ROOT_PATH = "F:\\SSHSite\\Samples"


def main():
    TFs = ['ATF2', 'ATF3', 'BHLHE40', 'CEBPB', 'CTCF', 'EGR1', 'ELF1', 'EZH2', 'FOS',
           'GABPA', 'GATA2', 'GTF2F1', 'HDAC2', 'JUN', 'JUND', 'MAFK', 'MAX', 'MAZ',
           'MXI1', 'MYC', 'NRF1', 'RAD21', 'REST', 'RFX5', 'SIN3A', 'SMC3', 'SP1', 'SRF',
           'SUZ12', 'TAF1', 'TCF12', 'TEAD4', 'TCF7L2', 'USF1', 'USF2', 'YY1']
    for TF in TFs:
        file_path = "{}\\{}\\sequence\\{}.csv".format(ROOT_PATH, TF, TF)
        model = Train_SkipGram(mer_num=3, d_model=16)
        model.learn(samples_file_path_l=[file_path], cell='', tf=TF)

main()