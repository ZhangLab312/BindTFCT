import numpy as np


class HighOrderEncoding:

    @staticmethod
    def build_mapper(order):
        base_pair = ['A', 'C', 'G', 'T']
        mapper = list([''])

        for _ in range(order):
            mapper_previous = mapper.copy()
            for nucleotide_pre in range(len(mapper_previous)):
                for nucleotide_now in base_pair:
                    mapper.append(mapper_previous[nucleotide_pre] + nucleotide_now)

            for _ in range(len(mapper_previous)):
                mapper.pop(0)

        one_hot = np.eye(len(mapper), dtype=int)
        high_order_code = dict()
        for i in range(len(mapper)):
            high_order_code[mapper[i]] = list(one_hot[i, :])

        return high_order_code

    @staticmethod
    def covert(sequence, order, mapper):
        code = np.empty(shape=(len(sequence) - order + 1, 4 ** order))
        padding = np.zeros(shape=(4 ** order))

        for loc in range(len(sequence) - order + 1):
            code[loc] = mapper.get(sequence[loc: loc + order])

        lr_round_flag = 0
        for pad in range(order - 1):
            if lr_round_flag == 0:
                code = np.row_stack((padding, code))
                lr_round_flag = 1
            else:
                code = np.row_stack((code, padding))
                lr_round_flag = 0

        return code


class seq2vec:

    def __init__(self):
        self.k_mer = 3
        self.d_code = 16

    def mer2idx(self):
        base_pair = ['A', 'C', 'T', 'G']
        mapper = list([''])

        for _ in range(self.k_mer):
            mapper_previous = mapper.copy()
            for nucleotide_pre in range(len(mapper_previous)):
                for nucleotide_now in base_pair:
                    mapper.append(mapper_previous[nucleotide_pre] + nucleotide_now)

            for _ in range(len(mapper_previous)):
                mapper.pop(0)

        index = range(0, 4 ** self.k_mer)
        mer2idx = dict()
        for i in range(len(mapper)):
            mer2idx[mapper[i]] = index[i]

        return mer2idx

    def Tokenizer(self, sequence):
        seq_len = len(sequence)

        k_mer_nucleotide = list()
        for loc in range(seq_len - self.k_mer + 1):
            k_mer_nucleotide.append(sequence[loc: loc + self.k_mer])

        return k_mer_nucleotide

    def converT(self, sequence, embeddinG_PATH):
        mer_index = self.mer2idx()
        Token_sequence = self.Tokenizer(sequence=sequence)

        Token2index = [mer_index[n] for n in Token_sequence]
        embeddinG = np.load(embeddinG_PATH)

        Token_embeddinG = np.empty(shape=(len(sequence) - self.k_mer + 1, self.d_code))

        for n, index in enumerate(Token2index):
            Token_embeddinG[n] = embeddinG[index]

        FlaG = 0
        paddinG = np.zeros(shape=self.d_code)
        for _ in range(self.k_mer - 1):
            if FlaG == 0:
                Token_embeddinG = np.row_stack((Token_embeddinG, paddinG))
                FlaG = 1
            else:
                Token_embeddinG = np.row_stack((paddinG, Token_embeddinG))
                FlaG = 0

        return Token_embeddinG