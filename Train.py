import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import random_split
from Samples.script import SSDataset_690
from Utils.EarlyStopping import EarlyStopping
from Methods.SSHSite import SSHSite


class Trainer:

    def __init__(self, model, TF, model_name='SSHSite', domain=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.TF = TF
        self.model_name = model_name
        self.domain = domain
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()
        self.domain_loss = nn.CrossEntropyLoss()

        self.batch_size = 64
        self.epochs = 10
        self.seq_dim = 16
        self.cell_num = 5

    def learn(self, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedModels"
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, his, label, d_label = data

                cell, site = self.model(seq.to(self.device, dtype=torch.float),
                                        shape.to(self.device, dtype=torch.float),
                                        his.to(self.device, dtype=torch.float))

                loss_1 = self.domain_loss(cell, d_label.squeeze(1).long().to(self.device))
                loss_2 = self.loss_function(site, label.float().to(self.device))
                final_loss = 0.5 * loss_1 + 0.5 * loss_2
                ProgressBar.set_postfix(loss=loss_2.item())

                final_loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_his, valid_labels, valid_d_labels in ValidateLoader:
                    valid_cell, valid_site = self.model(valid_seq.to(self.device, dtype=torch.float),
                                                        valid_shape.to(self.device, dtype=torch.float),
                                                        valid_his.to(self.device, dtype=torch.float))
                    valid_labels = valid_labels.float().to(self.device)

                    valid_loss.append(self.loss_function(valid_site, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)

            early_stopping(valid_loss_avg, self.model,
                           path + '\\' + self.TF + '.pth')

        print('\n---Finish Learn---\n')

    def learn_domain(self, domainLoader, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedModels"
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()

            """===== domain ====="""
            d_ProgressBar = tqdm(domainLoader)
            for d_data in d_ProgressBar:
                self.optimizer.zero_grad()

                d_ProgressBar.set_description("domain Epoch %d" % epoch)
                d_seq, d_shape, d_his, d_label = d_data
                d_output = self.model(d_seq.to(self.device, dtype=torch.float),
                                      d_his.to(self.device, dtype=torch.float),
                                      domain=True)
                d_loss = self.domain_loss(d_output, d_label.squeeze(1).long().to(self.device))
                d_ProgressBar.set_postfix(loss=d_loss.item())

                d_loss.backward()
                self.optimizer.step()

            """==== private ===="""
            p_ProgressBar = tqdm(TrainLoader)
            for p_data in p_ProgressBar:
                self.optimizer.zero_grad()

                p_ProgressBar.set_description("private Epoch %d" % epoch)
                p_seq, p_shape, p_his, p_label = p_data

                p_output = self.model(p_seq.to(self.device, dtype=torch.float),
                                      p_his.to(self.device, dtype=torch.float),
                                      domain=False)

                p_loss = self.loss_function(p_output, p_label.float().to(self.device))
                p_ProgressBar.set_postfix(loss=p_loss.item())

                p_loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_his, valid_labels in ValidateLoader:
                    valid_output = self.model(valid_seq.to(self.device, dtype=torch.float),
                                              valid_his.to(self.device, dtype=torch.float),
                                              domain=False)
                    valid_labels = valid_labels.float().to(self.device)

                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)

            early_stopping(valid_loss_avg, self.model,
                           path + '\\' + self.TF + '.pth')

        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):
        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedModels"
        # path = 'F:\\xiaorong\\SSHSite_shapeSavedModels'

        self.model.load_state_dict(torch.load(path + '\\' +
                                              self.TF + '.pth', map_location='cpu'))
        self.model.to("cpu")

        predicted_value = []
        ground_label = []
        self.model.eval()

        for seq, shape, his, label, d_labels in TestLoader:
            cell, site = self.model(seq.float(), shape.float(), his.float())
            # output = self.model(seq.float(), his.float(), True)
            """ To scalar"""
            # predicted_value.append(torch.argmax(cell.squeeze(dim=0)).detach().numpy())
            # ground_label.append(d_labels.squeeze(dim=0).squeeze(dim=0).detach().numpy())
            predicted_value.append(site.squeeze(dim=0).squeeze(dim=0).detach().numpy())
            ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().numpy())

        print('\n---Finish Inference---\n')

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        """
        accuracy = accuracy_score(y_true=ground_label, y_pred=predicted_value)
        kappa = cohen_kappa_score(y1=ground_label, y2=predicted_value)
        f_score = f1_score(y_true=ground_label, y_pred=predicted_value, average='macro')
        """
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=ground_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=ground_label)
        pr_auc = auc(recall, precision)

        f_score = f1_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)

        print('\n---Finish Measure---\n')

        return accuracy, roc_auc, pr_auc, f_score
        # return accuracy, kappa, f_score

    def save_evaluation_indicators(self, indicators):
        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedIndicators"

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = path + "\\" + self.model_name + "Indicators.xlsx"
        file = open(file_name, "a")

        file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + " " +
                   str(np.round(indicators[4], 4)) + "\n")
        """
        file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + "\n")
        """
        file.close()

    def run(self, samples_file_name, ratio=0.8):

        Train_Validate_Set = SSDataset_690(samples_file_name, self.seq_dim, False)

        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, self.seq_dim, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        self.learn(TrainLoader, ValidateLoader)

        predicted_value, ground_label = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc, f_score = self.measure(predicted_value, ground_label)
        # accuracy, kappa, f_score = self.measure(predicted_value, ground_label)

        # 写入评价指标
        # indicators = [self.TF, accuracy, kappa, f_score]
        indicators = [self.TF, accuracy, roc_auc, pr_auc, f_score]
        self.save_evaluation_indicators(indicators)

        print('\n---Finish Run---\n')

        # return accuracy, kappa, f_score


def main():
    TFs = ['ATF2', 'ATF3', 'BHLHE40', 'CEBPB', 'CTCF', 'EGR1', 'ELF1', 'EZH2', 'FOS',
           'GABPA', 'GATA2', 'GTF2F1', 'HDAC2', 'JUN', 'JUND', 'MAFK', 'MAX', 'MAZ',
           'MXI1', 'MYC', 'NRF1', 'RAD21', 'REST', 'RFX5', 'SIN3A', 'SMC3', 'SP1', 'SRF',
           'SUZ12', 'TAF1', 'TCF12', 'TEAD4', 'TCF7L2', 'USF1', 'USF2', 'YY1']

    for TF in TFs:
        Train = Trainer(model=SSHSite(seq_dim=16, shape_dim=4, h_m_dim=8),
                        TF=TF, model_name='SSHSite_new', domain=False)
        Train.run(samples_file_name=TF)


main()
"""

Train = Trainer(model=SSHSite(seq_dim=16, shape_dim=4, h_m_dim=8),
                TF='ATF2', model_name='SSHSite', domain=False)
Train.run(samples_file_name='ATF2')
"""
