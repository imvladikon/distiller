# http://www.kernel-operations.io/geomloss/index.html
# https://github.com/lxk00/BERT-EMD


from typing import Tuple

import torch
from torch import FloatTensor, nn
from torch.nn import functional as F
import numpy as np
import torch
from pyemd import emd_with_flow

from bert_email_labeling import logger
from utils import dotdict


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class AttentionEmdLoss(nn.Module):
    def __init__(self,
                 att_student_weight,
                 att_teacher_weight,
                 rep_student_weight,
                 rep_teacher_weight,
                 temperature: float = 1.,
                 device: str = None,
                 args=None):

        super().__init__()
        self.temperature = temperature
        self.att_student_weight = att_student_weight
        self.att_teacher_weight = att_teacher_weight
        self.rep_student_weight = rep_student_weight
        self.rep_teacher_weight = rep_teacher_weight
        for k in ["use_att", "update_weight", "use_rep", "embedding_emd", "separate", "add_softmax"]:
            assert k in args
        self.args = dotdict(args)
        self.T = temperature
        self.device = device
        self.loss_mse_fn = nn.MSELoss()

    def get_new_layer_weight(self, trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T, weights,
                             type_update='att'):
        if type_update == 'att':
            student_layer_weight = np.copy(self.att_student_weight)
            teacher_layer_weight = np.copy(self.self.att_teacher_weight)
        else:
            student_layer_weight = np.copy(self.self.rep_student_weight)
            teacher_layer_weight = np.copy(self.self.rep_teacher_weight)

        distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
        trans_weight = np.sum(trans_matrix * distance_matrix, -1)
        # logger.info('student_trans_weight:{}'.format(trans_weight))
        # new_student_weight = torch.zeros(stu_layer_num).to(device)
        for i in range(stu_layer_num):
            student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
        weight_sum = np.sum(student_layer_weight)
        for i in range(stu_layer_num):
            if student_layer_weight[i] != 0:
                student_layer_weight[i] = weight_sum / student_layer_weight[i]

        trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
        for j in range(tea_layer_num):
            teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
        weight_sum = np.sum(teacher_layer_weight)
        for i in range(tea_layer_num):
            if teacher_layer_weight[i] != 0:
                teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]

        student_layer_weight = softmax(student_layer_weight / T)
        teacher_layer_weight = softmax(teacher_layer_weight / T)

        if type_update == 'att':
            self.att_student_weight = student_layer_weight
            self.self.att_teacher_weight = teacher_layer_weight
        else:
            self.self.rep_student_weight = student_layer_weight
            self.self.rep_teacher_weight = teacher_layer_weight

    def transformer_loss(self,
                         student_atts,
                         teacher_atts,
                         student_reps,
                         teacher_reps,
                         device):

        def embedding_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                               stu_layer_num, tea_layer_num):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            totol_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([totol_num, totol_num]).to(device)
            for i in range(stu_layer_num):
                student_rep = student_reps[i]
                for j in range(tea_layer_num):
                    teacher_rep = teacher_reps[j]
                    tmp_loss = self.loss_mse_fn(student_rep, teacher_rep)
                    # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            # trans_matrix = trans_matrix
            rep_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)
            return rep_loss, trans_matrix, distance_matrix

        def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                         stu_layer_num, tea_layer_num):
            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            totol_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([totol_num, totol_num]).to(device)
            for i in range(stu_layer_num):
                student_rep = student_reps[i + 1]
                for j in range(tea_layer_num):
                    teacher_rep = teacher_reps[j + 1]
                    tmp_loss = self.loss_mse_fn(student_rep, teacher_rep)
                    # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            # trans_matrix = trans_matrix
            rep_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)
            return rep_loss, trans_matrix, distance_matrix

        def emd_att_loss(student_atts, teacher_atts, student_layer_weight, teacher_layer_weight,
                         stu_layer_num, tea_layer_num, device):

            student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
            teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
            totol_num = stu_layer_num + tea_layer_num
            distance_matrix = torch.zeros([totol_num, totol_num]).to(device)
            for i in range(stu_layer_num):
                student_att = student_atts[i]
                for j in range(tea_layer_num):
                    teacher_att = teacher_atts[j]
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)

                    #TODO: fix a bug, when different count of the attention heads
                    tmp_loss = self.loss_mse_fn(student_att, teacher_att)
                    distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
            _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                            distance_matrix.detach().cpu().numpy().astype('float64'))
            att_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)
            return att_loss, trans_matrix, distance_matrix

        stu_layer_num = len(student_atts)
        tea_layer_num = len(teacher_atts)
        if self.args.use_att:
            att_loss, att_trans_matrix, att_distance_matrix = \
                emd_att_loss(student_atts, teacher_atts, self.att_student_weight, self.att_teacher_weight,
                             stu_layer_num, tea_layer_num, device)
            if self.args.update_weight:
                self.get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
            att_loss = att_loss.to(device)
        else:
            att_loss = torch.tensor(0)
        if self.use_rep:
            if self.args.embedding_emd:
                rep_loss, rep_trans_matrix, rep_distance_matrix = \
                    embedding_rep_loss(student_reps, teacher_reps, self.rep_student_weight, self.rep_teacher_weight,
                                       stu_layer_num + 1, tea_layer_num + 1, device)
                if self.args.update_weight:
                    self.get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num + 1,
                                              tea_layer_num + 1,
                                              T=self.T,
                                              type_update='xx')
            else:
                rep_loss, rep_trans_matrix, rep_distance_matrix = \
                    emd_rep_loss(student_reps, teacher_reps, self.rep_student_weight, self.rep_teacher_weight,
                                 stu_layer_num, tea_layer_num)

                if self.args.update_weight:
                    self.get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num,
                                              T=self.T,
                                              type_update='xx')
            rep_loss = rep_loss.to(device)
        else:
            rep_loss = torch.tensor(0)

        if not self.args.separate:
            student_weight = np.mean(np.stack([self.att_student_weight, self.rep_student_weight]), 0)
            teacher_weight = np.mean(np.stack([self.att_teacher_weight, self.rep_teacher_weight]), 0)
            # if global_step % args.eval_step == 0:
            #     logger.info('all_student_weight:{}'.format(student_weight))
            #     logger.info('all_teacher_weight:{}'.format(teacher_weight))
            att_student_weight = student_weight
            self.att_teacher_weight = teacher_weight
            self.rep_student_weight = student_weight
            self.rep_teacher_weight = teacher_weight
        else:
            pass
            # if global_step % args.eval_step == 0:
            #     logger.info('att_student_weight:{}'.format(att_student_weight))
            #     logger.info('self.att_teacher_weight:{}'.format(self.att_teacher_weight))
            #     logger.info('self.rep_student_weight:{}'.format(self.rep_student_weight))
            #     logger.info('self.rep_teacher_weight:{}'.format(self.rep_teacher_weight))
        if self.args.add_softmax:
            self.att_student_weight = softmax(self.att_student_weight)
            self.self.att_teacher_weight = softmax(self.att_teacher_weight)

            self.rep_student_weight = softmax(self.rep_student_weight)
            self.rep_teacher_weight = softmax(self.rep_teacher_weight)

        return att_loss, rep_loss

    def forward(
            self,
            s_hidden_states: Tuple[FloatTensor],
            s_attentions: Tuple[FloatTensor],
            t_hidden_states: Tuple[FloatTensor],
            t_attentions: Tuple[FloatTensor],
    ) -> FloatTensor:

        att_loss, rep_loss = self.transformer_loss(s_attentions,
                                                   t_attentions,
                                                   s_hidden_states,
                                                   t_hidden_states,
                                                   self.device)
        return att_loss
