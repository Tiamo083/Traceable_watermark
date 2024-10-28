from statistics import mean
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, train_config):
        super(Loss, self).__init__()
        self.msg_loss = nn.MSELoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg)
        return embedding_loss, msg_loss

class Loss_identity(nn.Module):
    def __init__(self, train_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Loss_identity, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
        self.fragile_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.fragile_target = torch.tensor(0.5).to(device)
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        attack_msg_loss = self.msg_loss(msg, rec_msg[0]) 
        no_attack_msg_loss = self.msg_loss(msg, rec_msg[1])
        return embedding_loss, no_attack_msg_loss, attack_msg_loss

    def half_en_de_loss(self, x, w_x, msg, att_rec_msg, no_att_rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        robust_msg, fragile_msg = torch.chunk(input = msg, chunks = 2, dim = 2)
        rec_r_msg, rec_f_msg = torch.chunk(input = att_rec_msg, chunks = 2, dim = 2)
        attack_r_loss = self.msg_loss(robust_msg, rec_r_msg)
        attack_f_loss = self.fragile_loss(fragile_msg, rec_f_msg)
        # import pdb
        # pdb.set_trace()
        real_f_loss = self.msg_loss(attack_f_loss, self.fragile_target)
        no_attack_loss = self.msg_loss(no_att_rec_msg, msg)

        return embedding_loss, no_attack_loss, attack_r_loss, real_f_loss

    def multi_de_loss(self, x, w_x, msg, attack_rec_robust_msg, no_attack_rec_robust_msg, attack_rec_fragile_msg, no_attack_rec_fragile_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        robust_msg, fragile_msg = torch.chunk(input = msg, chunks = 2, dim = 2)
        attack_r_loss = self.msg_loss(robust_msg, attack_rec_robust_msg)
        no_attack_r_loss = self.msg_loss(robust_msg, no_attack_rec_robust_msg)
        
        attack_f_loss = self.msg_loss(fragile_msg, attack_rec_fragile_msg)
        no_attack_f_loss = self.msg_loss(fragile_msg, no_attack_rec_fragile_msg)
        
        return embedding_loss, no_attack_r_loss, attack_r_loss, no_attack_f_loss, attack_f_loss


class Loss_identity_3(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_3, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
        # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
        msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2]) + self.msg_loss(msg, rec_msg[3])
        return embedding_loss, msg_loss

class Loss_identity_3_2(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_3_2, self).__init__()
        # self.msg_loss = nn.MSELoss()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
        # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
        msg_loss =  self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[2].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[3].squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Loss2(nn.Module):
    def __init__(self, train_config):
        super(Loss2, self).__init__()
        # self.msg_loss = nn.MSELoss()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(rec_msg.squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Loss_identity_2(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_2, self).__init__()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Lossex(nn.Module):
    def __init__(self, train_config):
        super(Lossex, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg, no_msg, no_decoded):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg)
        no_msg_loss = self.msg_loss(no_msg, no_decoded)
        return embedding_loss, msg_loss, no_msg_loss
