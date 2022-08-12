import torch
import torch.nn as nn


class SSHSite(nn.Module):

    def __init__(self, seq_dim, shape_dim, h_m_dim):
        super(SSHSite, self).__init__()
        self.SHARE_EXPERT = EXPERT_NET(seq_dim, shape_dim, h_m_dim)

        self.TFBS_EXPERT = EXPERT_NET(seq_dim, shape_dim, h_m_dim)
        self.CELL_EXPERT = EXPERT_NET(seq_dim, shape_dim, h_m_dim)

        self.TASK_GATE_TFBS = TASK_GATE()
        self.TASK_GATE_CELL = TASK_GATE()

        self.CGC_NET = CGC_NET()

        self.AITTM = AITTM(i_dim=24)

    def forward(self, seq, shape, h_m):
        o_SHARE = self.SHARE_EXPERT(seq, shape, h_m)

        w_CELL = self.TASK_GATE_CELL(seq, shape, h_m)
        o_CELL = self.CELL_EXPERT(seq, shape, h_m)
        o_CELL = self.CGC_NET(o_SHARE, o_CELL, w_CELL)

        w_TFBS = self.TASK_GATE_TFBS(seq, shape, h_m)
        o_TFBS = self.TFBS_EXPERT(seq, shape, h_m)
        o_TFBS = self.CGC_NET(o_SHARE, o_TFBS, w_TFBS)

        return self.AITTM(o_CELL, o_TFBS)


class ms_sam(nn.Module):

    def __init__(self, kernel_size, pad):
        super(ms_sam, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, 1, pad)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        sam_map = torch.cat((avg_pool, max_pool), 1)
        sam_map = self.conv(sam_map)
        sam_map = self.sigmoid(sam_map)

        return x * sam_map


class conv(nn.Module):

    def __init__(self, i_dim, o_dim, kernel_size, pad, sam=True):
        super(conv, self).__init__()
        self.bn_conv = nn.Sequential(
            nn.Conv1d(in_channels=i_dim,
                      out_channels=o_dim,
                      kernel_size=kernel_size,
                      padding=pad),
            nn.BatchNorm1d(num_features=o_dim),
            nn.ReLU(inplace=True)
        )

        self.sam = sam

        if self.sam:
            self.ms_sam = ms_sam(kernel_size, pad)

    def forward(self, x):
        if self.sam:
            y = self.ms_sam(self.bn_conv(x))
        else:
            y = self.bn_conv(x)
        return y


class ms_conv_sq(nn.Module):

    def __init__(self, i_dim):
        super(ms_conv_sq, self).__init__()

        self.conv_1 = conv(i_dim, 24, 23, 11, True)

        self.conv_2 = conv(i_dim, 24, 11,  5, True)

        self.conv_3 = conv(i_dim, 24,  7,  3, True)

    def forward(self, x):
        x_branch_1 = self.conv_1(x)
        x_branch_2 = self.conv_2(x)
        x_branch_3 = self.conv_3(x)

        return x_branch_1, x_branch_2, x_branch_3


class ms_conv_sh(nn.Module):

    def __init__(self, i_dim):
        super(ms_conv_sh, self).__init__()

        self.conv_1 = conv(i_dim, 24, 21, 10, False)

        self.conv_2 = conv(i_dim, 24,  9,  4, False)

        self.conv_3 = conv(i_dim, 24,  3,  1, False)

    def forward(self, x):
        x_branch_1 = self.conv_1(x)
        x_branch_2 = self.conv_2(x)
        x_branch_3 = self.conv_3(x)

        return x_branch_1, x_branch_2, x_branch_3


class ms_conv_hm(nn.Module):

    def __init__(self, i_dim):
        super(ms_conv_hm, self).__init__()

        self.conv_1 = conv(i_dim, 24, 5, 2, False)

        self.conv_2 = conv(i_dim, 24, 3, 1, False)

        self.conv_3 = conv(i_dim, 24, 1, 0, False)

    def forward(self, x):
        x_branch_1 = self.conv_1(x)
        x_branch_2 = self.conv_2(x)
        x_branch_3 = self.conv_3(x)

        return x_branch_1, x_branch_2, x_branch_3


class TASK_GATE(nn.Module):

    def __init__(self):
        super(TASK_GATE, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1)
        )

        self.w_GATE = nn.Sequential(
            nn.Linear(in_features=28, out_features=18),
            nn.Softmax(dim=1)
        )

    def forward(self, seq, shp, h_m):
        x = torch.cat((self.pool(seq), self.pool(shp), self.pool(h_m)), 1)

        w = self.w_GATE(x)

        return w


class CROSS_GATE(nn.Module):

    def __init__(self, i_dim):
        super(CROSS_GATE, self).__init__()
        self.GATE = nn.Sequential(
            nn.Conv1d(in_channels=2 * i_dim,
                      out_channels=1,
                      kernel_size=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=2)
        )

    def forward(self, seq, shape_or_hm):

        w = self.GATE(torch.cat((seq, shape_or_hm), 1))

        y = shape_or_hm * w

        return y


class EXPERT_NET(nn.Module):

    def __init__(self, seq_dim, shape_dim, h_m_dim):
        super(EXPERT_NET, self).__init__()
        self.conv_seq = ms_conv_sq(i_dim=seq_dim)
        self.conv_shp = ms_conv_sh(i_dim=shape_dim)
        self.conv_h_m = ms_conv_hm(i_dim=h_m_dim)

        self.CROSS_GATE_HM = CROSS_GATE(i_dim=24)
        self.CROSS_GATE_SP = CROSS_GATE(i_dim=24)

        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(1)
        )

    def forward(self, seq, shp, h_m):
        mcsq_1, mcsq_2, mcsq_3 = self.conv_seq(seq)
        mcsp_1, mcsp_2, mcsp_3 = self.conv_shp(shp)
        mchm_1, mchm_2, mchm_3 = self.conv_h_m(h_m)

        mchm_1 = self.CROSS_GATE_HM(mcsq_1, mchm_1)
        mchm_2 = self.CROSS_GATE_HM(mcsq_2, mchm_2)
        mchm_3 = self.CROSS_GATE_HM(mcsq_3, mchm_3)

        mcsp_1 = self.CROSS_GATE_SP(mcsq_1, mcsp_1)
        mcsp_2 = self.CROSS_GATE_SP(mcsq_2, mcsp_2)
        mcsp_3 = self.CROSS_GATE_SP(mcsq_3, mcsp_3)

        mcsq_1, mcsq_2, mcsq_3 = self.pool(mcsq_1), self.pool(mcsq_2), self.pool(mcsq_3)
        mchm_1, mchm_2, mchm_3 = self.pool(mchm_1), self.pool(mchm_2), self.pool(mchm_3)
        mcsp_1, mcsp_2, mcsp_3 = self.pool(mcsp_1), self.pool(mcsp_2), self.pool(mcsp_3)

        return [mcsq_1, mcsq_2, mcsq_3, mcsp_1, mcsp_2, mcsp_3, mchm_1, mchm_2, mchm_3]


class CGC_NET(nn.Module):

    def __init__(self):
        super(CGC_NET, self).__init__()

    def forward(self, s, p, w_GATE):
        """
        s: share
        p: private
        """
        w_GATE = torch.transpose(w_GATE, 0, 1).unsqueeze(2).expand(-1, -1, 24)

        o_EXPT = torch.stack(s + p)

        rec_o_EXPT = o_EXPT * w_GATE

        y = torch.sum(rec_o_EXPT, dim=0)

        return y


"""
============= AITTM ============
"""


class Tower(nn.Module):
    def __init__(self, i_dim, r=2):
        super(Tower, self).__init__()

        self.layer = nn.Sequential(nn.Linear(i_dim, i_dim // r),
                                   nn.ReLU(True))

    def forward(self, x):
        return self.layer(x)


class Attention(nn.Module):
    def __init__(self, i_dim):
        super(Attention, self).__init__()
        self.dim = i_dim

        self.q_layer = nn.Linear(i_dim, i_dim, bias=False)
        self.k_layer = nn.Linear(i_dim, i_dim, bias=False)
        self.v_layer = nn.Linear(i_dim, i_dim, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)

        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)

        return torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)


class AITTM(nn.Module):

    def __init__(self, i_dim, r=2, c=5):
        super(AITTM, self).__init__()
        self.cell_layer = Tower(i_dim=i_dim, r=r)
        self.site_layer = Tower(i_dim=i_dim, r=r)

        self.info_layer = nn.Sequential(nn.Linear(i_dim // r, i_dim // r),
                                        nn.ReLU(True))

        self.attention_layer = Attention(i_dim=(i_dim // r))

        self.cell = nn.Linear(i_dim // r, c)

        self.site = nn.Sequential(nn.Linear(i_dim // r, 1),
                                  nn.Sigmoid())

    def forward(self, x_cell, x_site):
        x_site = self.site_layer(x_site)

        x_info = torch.unsqueeze(self.info_layer(x_site), dim=1)
        x_cell = torch.unsqueeze(self.cell_layer(x_cell), dim=1)

        x_attention = self.attention_layer(torch.cat((x_cell, x_info), dim=1))

        return self.cell(x_attention), self.site(x_site)