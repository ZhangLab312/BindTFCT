import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


def ShapeRToCsv(path, shapes, seq_len):
    """
        将DNAShapeR工具得到的shape样本，保存至csv格式的文件

        :parameters
            shapes是列表，其中存储待处理的shape类型的名称，如Roll等
            path的格式，需要文件名(如Test_shape.data)，但不需要.shape_name

        结果将保存在相同的文件夹中
        一次只能处理一个DataSet
    """
    for shape in shapes:
        i_file = open(path + '.' + shape)
        o_file = csv.writer(open(path + shape + '.csv', 'w', newline=''))

        # """
        #     write header
        # """
        # row = []
        # for i in range(seq_len):
        #     row.append(i + 1)
        #
        # for line in i_file.readlines():
        #     """
        #         文件格式:
        #             >1
        #             NA,NA,4.96,.......,4.92,NA,NA
        #     """
        #     line = line.replace('\n', '')
        #     if line[0] == '>':
        #         o_file.writerow(row)
        #         row = []
        #     else:
        #         line = line.split(',')
        #         for char in line:
        #             if char == 'NA':
        #                 row.append(float(0))
        #             else:
        #                 row.append(float(char))
        # ！！！！！！！！！！！！！！！！！！！！纠正  漏掉了最后一条数据，然后把索引去掉，不要索引了
        row = []
        for index_line, line in enumerate(i_file):
            if line[0] == ">":
                # print(line)
                words = line.replace("\n", "").split("_")
                # 碰到起始符就把上一条形状写进去
                if row:
                    if shape == "HelT" or shape == 'Roll':
                        row.insert(0, 0)
                        row.insert(len(row), 0)
                    o_file.writerow(row)
                    # 把row清空，为下一条序列做准备
                    row = []
            else:
                # print(line)
                line = line.replace("\n", "").split(',')
                for char in line:
                    if char == 'NA':
                        row.append(float(0))
                    else:
                        row.append(float(char))
            # # 写一条形状数据
            # if (index_line + 1) // 5 == int(words[-1]):
            #     # print(row)
            #     # 对于HelT和Roll这两种形状，在前后各添加一个0，拓展到102位
            #     if shape == "HelT" or shape == 'Roll':
            #         row.insert(0, 0)
            #         row.insert(len(row), 0)
            #     o_file.writerow(row)
            #     # 把row清空，为下一条序列做准备
            #     row = []
        # 写入最后一条数据
        if shape == "HelT" or shape == 'Roll':
            row.insert(0, 0)
            row.insert(len(row), 0)
        o_file.writerow(row)
    print('\033[32m' + 'success!')


def mean_HelT_Roll(HelT_p, Roll_p, TFs=None, cells=None):
    HelT = pd.read_csv(HelT_p, header=None)
    Roll = pd.read_csv(Roll_p, header=None)

    samples_num = HelT.shape[0]
    samples_len = HelT.shape[1]

    new_HelT = np.empty(shape=(samples_num, samples_len-1))
    new_Roll = np.empty(shape=(samples_num, samples_len-1))

    for i in range(samples_num):
        HelT_samples = HelT.loc[i]
        Roll_samples = Roll.loc[i]

        for k in range(0, samples_len - 1):
            HelT_samples_new = (HelT_samples[k] + HelT_samples[k + 1]) / 2
            Roll_samples_new = (Roll_samples[k] + Roll_samples[k + 1]) / 2

            new_HelT[i][k] = HelT_samples_new
            new_Roll[i][k] = Roll_samples_new

    new_HelT = np.round(new_HelT, decimals=2)
    new_Roll = np.round(new_Roll, decimals=2)

    o_HelT = csv.writer(open('new_HelT.csv', 'w', newline=''))
    o_Roll = csv.writer(open('new_Roll.csv', 'w', newline=''))

    o_HelT.writerows(new_HelT)
    o_Roll.writerows(new_Roll)


"""
CELL_TYPE = ["GM12878", "H1", "Hela-S3", "HepG2", "K562"]
# CELL_TYPE = ["HepG2"]
PATH_ROOT = "D:\\MRJOHN\\BHSITE2\\processed"
tfs = pd.read_excel("D:\\MRJOHN\\BHSITE2\\processed\\tfs.xlsx")
progress_bar = tqdm(CELL_TYPE)
for index_cell, cell in enumerate(progress_bar):
    progress_bar.set_description("{}".format(cell))
    for index_tf, tf in enumerate(tfs[cell].dropna()):
        path = PATH_ROOT + "\\{}\\sequence\\{}nSet.fa".format(cell, tf)
        ShapeRToCsv(path=path,
                    shapes=['HelT', 'MGW', 'ProT', 'Roll'], seq_len=101)
        path = PATH_ROOT + "\\{}\\sequence\\{}pSet.fa".format(cell, tf)
        ShapeRToCsv(path=path,
                    shapes=['HelT', 'MGW', 'ProT', 'Roll'], seq_len=101)
    #     break
    # break
"""
mean_HelT_Roll(HelT_p='D:\\PycharmProjects\\SSHSite\\Utils\\ATF3nSet.faHelT.csv',
               Roll_p='D:\\PycharmProjects\\SSHSite\\Utils\\ATF3nSet.faHelT.csv')
