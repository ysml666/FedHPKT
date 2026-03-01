import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clients import clienthpkt
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server


class FedHPKT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clienthpkt)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            PROTO = Trainable_prototypes(
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
            save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
            print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

    def train(self):
        global adversarial_protos
        self.done_flag = False
        l=0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()


            if i < (l + (self.global_rounds - l) / 2):
                var = 1 - (i - l) * (1 - 0.5) / ((self.global_rounds - l) / 2)
            else:
                var = 0.5

            print("var:", var)

            for client in self.selected_clients:
                client.train(var)

            self.receive_protos()
            self.update_Gen()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            self.save_results()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))


    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

    def update_Gen(self):
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        PROTO = PROTO.to(self.device)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()

        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                proto = proto.to(self.device)
                y = torch.as_tensor(y, dtype=torch.long, device=self.device)

                proto_gen = PROTO(torch.arange(self.num_classes, device=self.device))

                cos_sim = F.cosine_similarity(proto.unsqueeze(1), proto_gen.unsqueeze(0), dim=2)
                loss = function_loss(cos_sim, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)

        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

def function_loss(cos_sim, y):
    one_hot = F.one_hot(y, cos_sim.size(1)).float()

    positive_sim = torch.sum(one_hot * cos_sim, dim=1)

    negative_sim = (1 - one_hot) * cos_sim

    m = 1.0 - positive_sim

    n = torch.clamp(negative_sim - positive_sim.unsqueeze(1), min=0)
    n = torch.sum(n, dim=1)

    loss = torch.mean(m+n)

    return loss


class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out
