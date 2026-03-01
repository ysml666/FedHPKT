import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientProto145(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.temp = 25.0

    @staticmethod
    def slerp_batch(u, v, t=0.5, eps=1e-6):

        dot = (u * v).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp_min(eps)

        s1 = torch.sin((1 - t) * theta) / sin_theta
        s2 = torch.sin(t * theta) / sin_theta
        out = s1 * u + s2 * v
        return out

    def build_spa_logits(self, rep, y, protos_tensor, cos_sim_mat, rep_norm=None):

        device = self.device
        B = rep.size(0)

        if rep_norm is None:
            rep_norm = F.normalize(rep, dim=1)

        sim_mask = cos_sim_mat.clone()
        sim_mask[torch.arange(B, device=device), y] = -1e9
        hard_neg = sim_mask.argmax(dim=1)  # (B,)

        P_pos = protos_tensor[y]           # (B, D)
        P_neg = protos_tensor[hard_neg]    # (B, D)

        with torch.no_grad():
            P_pos_n = F.normalize(P_pos, dim=1)
            P_neg_n = F.normalize(P_neg, dim=1)
            P_adv_pos = self.slerp_batch(P_pos_n, P_neg_n, t=0.5)  # (B, D), unit-ish

        logits_spa = cos_sim_mat.clone()
        adv_pos_sim = (rep_norm * P_adv_pos).sum(dim=1)  # (B,)
        logits_spa[torch.arange(B, device=device), y] = adv_pos_sim

        return logits_spa, hard_neg

    def train(self, v):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)

        protos_tensor = None
        if global_protos is not None:
            keys = sorted(global_protos.keys())
            protos_tensor = torch.stack([global_protos[k].to(self.device) for k in keys], dim=0)  # (C, D)

        for step in range(max_local_epochs):
            for batch_idx, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if protos_tensor is not None:
                    rep_norm = F.normalize(rep, dim=1)  # (B, D)
                    features_into_centers = torch.matmul(rep, protos_tensor.T)
                    proto_norms = torch.norm(rep, dim=1, keepdim=True)
                    proto_gen_norms = torch.norm(protos_tensor, dim=1, keepdim=True)
                    norm_product = torch.matmul(proto_norms, proto_gen_norms.T)
                    cos_sim_mat = features_into_centers / norm_product

                    dynamic_weight = torch.sigmoid(cos_sim_mat.mean(dim=1, keepdim=True))  # (B, 1)

                    logits_spg = cos_sim_mat * dynamic_weight
                    loss_spg = self.loss(logits_spg * self.temp, y)
                    loss = loss + loss_spg * v

                    if v != 1:
                        logits_spa, hard_neg = self.build_spa_logits(
                            rep=rep,
                            y=y,
                            protos_tensor=protos_tensor,
                            cos_sim_mat=cos_sim_mat,
                            rep_norm=rep_norm
                        )
                        dynamic_weight1 = torch.sigmoid(logits_spa.mean(dim=1, keepdim=True))

                        logits_spa = logits_spa * dynamic_weight1
                        loss_spa = self.loss(logits_spa * self.temp, y)
                        loss = loss + loss_spa  * (1 - v)

                for j, yy in enumerate(y):
                    y_c = int(yy.item())
                    protos[y_c].append(rep[j, :].detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(self.agg_func(protos, global_protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        correct_num = 0
        test_num = 0

        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    output = torch.zeros(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                cos_sim = F.cosine_similarity(r.unsqueeze(0), pro.unsqueeze(0))
                                output[i, j] = cos_sim

                    correct_num += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return correct_num, test_num
        else:
            return 0, 1e-5


    def agg_func(self, protos, global_protos):

        if global_protos is None:
            out = defaultdict(list)
            for label, proto_list in protos.items():
                proto_mat = torch.stack([p for p in proto_list], dim=0)  # (n, D) on CPU
                out[label] = proto_mat.mean(dim=0)
            return out

        keys = sorted(global_protos.keys())
        global_tensor = torch.stack([global_protos[k].detach().to(self.device) for k in keys], dim=0)  # (C, D)
        global_norm = F.normalize(global_tensor, dim=1)  # (C, D)

        out = defaultdict(list)

        for label, proto_list in protos.items():
            if label not in global_protos:
                proto_mat = torch.stack([p for p in proto_list], dim=0)
                out[label] = proto_mat.mean(dim=0)
                continue

            proto_mat_cpu = torch.stack([p for p in proto_list], dim=0)  # (n, D) on CPU
            proto_mat = proto_mat_cpu.to(self.device)                    # (n, D)
            proto_norm = F.normalize(proto_mat, dim=1)                   # (n, D)

            cos_sim = proto_norm @ global_norm.t()

            # target_sim: (n,)
            target_sim = cos_sim[:, label]

            cos_sim_excl = cos_sim.clone()
            cos_sim_excl[:, label] = -1e9
            max_other_sim = cos_sim_excl.max(dim=1).values

            keep_mask = target_sim > max_other_sim
            if keep_mask.any():
                kept = proto_mat_cpu[keep_mask.detach().cpu()]
                out[label] = kept.mean(dim=0)
            else:
                best_idx = target_sim.argmax().item()
                out[label] = proto_mat_cpu[best_idx]

        return out