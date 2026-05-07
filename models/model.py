import logging
from math import e
import numpy as np
import torch
import os
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from utils.data_manager import DummyDataset
from utils.inc_net import SimpleVitNet, HashingLayer
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')



class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8

        self.num_workers = args['num_workers'] if args['num_workers'] is not None else 4

        self.args = args
        
        self.hashing_layer = None

        self._feature_class_protos = []  
        self._hash_protos = []  

        self._covs = []  
        self._crs_cor = []
        self._covs_proj = []

        self.projections = []

    def _inverse_fp64(self, mat):

        out_dtype = mat.dtype
        m64 = mat.to(device=self._device, dtype=torch.float64)
        inv64 = torch.linalg.inv(m64)
        return inv64.to(dtype=out_dtype)

    def after_task(self, inc=False):
        self._known_classes = self._total_classes
        self._network.backbone.add_adapter_to_list()

        self._projector_loss = self._get_projector(self.hashing_layer.G, self.args["explained_variance_ratio"])


    def incremental_train(self, data_manager):
        self.data_manager = data_manager

        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.init_fc(self._hash_code_length)

        if self.hashing_layer is not None:
            with torch.no_grad():
                R_inv = self._inverse_fp64(self.hashing_layer.G + self.hashing_layer.R)
                Delta_hash = R_inv @ self.hashing_layer.Q
                self._network.fc.weight.copy_(torch.t(Delta_hash.float()))
            logging.info("FC layer initialized with Delta_hash from hashing_layer")

        if self.hashing_layer is None:
            self.hashing_layer = HashingLayer(self._network.feature_dim, self._hash_code_length, self._device, self.args)
            for name, param in self.hashing_layer.named_parameters():
                param.requires_grad = False
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )        
        logging.info(
            "hash code length {}".format(self._hash_code_length)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        #self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), 
            source="test", 
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']
        if self._cur_task == 0:
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = self._get_optimizer(lr=self.args["init_lr"])
                scheduler = self._get_scheduler(optimizer, self.args["tuned_epoch"])

                self._init_train(train_loader, test_loader, optimizer, scheduler)
                del optimizer, scheduler
                self._network.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            train_dataset =self.data_manager.get_dataset(
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    mode="test",
                )

            train_loader = DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
                )

            self._network.eval()

            cov = torch.zeros(self._network.feature_dim, self._network.feature_dim).to(self._device)
            crs_cor = torch.zeros(self._network.feature_dim, self._hash_code_length).to(self._device)
            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    output = self._network(inputs)
                    out_features, out_hashing_codes = output["features"], output["hash_code"]
                    cov += torch.t(out_features) @ out_features
                    crs_cor += torch.t(out_features) @ (out_hashing_codes)

            self.hashing_layer.G = cov
            self.hashing_layer.Q = crs_cor


            self._covs.append(cov.cpu())
            self._crs_cor.append(crs_cor.cpu())

            projector = self._get_projector(cov, self.args["explained_variance_ratio"])
            self._covs_proj.append(projector)

            self._build_database_hash_code_and_protos() 

            
        else:
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = self._get_optimizer(lr=self.args["init_lr"])
                scheduler = self._get_scheduler(optimizer, self.args["tuned_epoch"])

                self._update_representation(train_loader, test_loader, optimizer, scheduler)
                del optimizer, scheduler
                self._network.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            train_dataset =self.data_manager.get_dataset(
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    mode="test",
                )
            train_loader = DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
                )
            
            self._network.eval()
            feat_dim = self._network.feature_dim
            cov_t = torch.zeros(feat_dim, feat_dim).to(self._device)
            crs_cor_t = torch.zeros(feat_dim, self._hash_code_length).to(self._device)

            cov_proj = [self.args["rg"] * torch.eye(feat_dim).to(self._device) for _ in range(self._cur_task)]
            crs_proj = [torch.zeros(feat_dim, feat_dim).to(self._device) for _ in range(self._cur_task)]

            with torch.no_grad():
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)

                    output_t = self._network(inputs)
                    feats_t = output_t["features"]
                    hash_codes_t = output_t["hash_code"]

                    cov_t += feats_t.t() @ feats_t
                    crs_cor_t += feats_t.t() @ hash_codes_t

                    for i in range(self._cur_task):
                        feats_i = self._network.backbone.forward_cumulative(inputs, adapter_index=i)
 
                        cov_proj[i] += feats_i.t() @ feats_i
 
                        crs_proj[i] += feats_i.t() @ (feats_t-feats_i)
 

            self._covs.append(cov_t.cpu())
            self._crs_cor.append(crs_cor_t.cpu())
            projector = self._get_projector(cov_t, self.args["explained_variance_ratio"])
            self._covs_proj.append(projector)

            lambda_ema = self.args["lambda_ema"]

            R_inv_last = self._inverse_fp64(cov_proj[-1])
            P_last = R_inv_last @ crs_proj[-1] + torch.eye(self._network.feature_dim).to(self._device)
            self.projections.append(P_last)

            for i in range(self._cur_task-1):
                R_inv = self._inverse_fp64(cov_proj[i])
                P_i = R_inv @ crs_proj[i] + torch.eye(self._network.feature_dim).to(self._device)
                self.projections[i] = lambda_ema * P_i + (1-lambda_ema) * self.projections[i] @ self.projections[-1]
                self._computer_error(self.projections[i], i)
                
            self._computer_error(self.projections[-1], self._cur_task-1)

            del cov_proj, crs_proj
            torch.cuda.empty_cache()

            G = cov_t.clone()
            Q = crs_cor_t.clone()

            for i in range(self._cur_task):
                P_i = self.projections[i]
                covs_i = self._covs[i].to(self._device)
                crs_cor_i = self._crs_cor[i].to(self._device)
                G += P_i.t() @ covs_i @ P_i
                Q += P_i.t() @ crs_cor_i


            self.hashing_layer.G = G
            self.hashing_layer.Q = Q

            R_inv = self._inverse_fp64(G + self.hashing_layer.R)
            Delta_hash = R_inv @ Q
            self.hashing_layer.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta_hash.float()))

            self._build_database_hash_code_and_protos()

            self._network.update_fc(self.hashing_layer)


    def _computer_error(self, Delta, cur_task):
        
        if cur_task == 0:
            start = 0
            end = self.args['init_cls']
        else:
            start = self.args['increment'] * (cur_task-1) + self.args['init_cls']
            end = self.args['increment'] + start

        train_dataset_old =self.data_manager.get_dataset(
                np.arange(start, end),
                source="train",
                mode="test",
            )
        train_loader_old = DataLoader(
                train_dataset_old, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
        sq_sum_update = torch.tensor(0.0, device=self._device)
        sq_sum_orig = torch.tensor(0.0, device=self._device)
       
        num_samples = 0

        with torch.no_grad():
            for index, inputs, _ in train_loader_old:
                inputs = inputs.to(self._device)
                feats_from = self._network.backbone.forward_cumulative(inputs, adapter_index=cur_task)
                feats_to = self._network.backbone.forward_cumulative(inputs, adapter_index=self._cur_task)
                sq_sum_update += ((feats_from @ Delta - feats_to) ** 2).sum()
                sq_sum_orig += ((feats_from - feats_to) ** 2).sum()
                num_samples += inputs.size(0)

        error_update = (sq_sum_update / num_samples).sqrt()
        error = (sq_sum_orig / num_samples).sqrt()
        logging.info("Projection (class {}->{}): Error of update: {:.4f}, Error of original: {:.4f}".format(
            start, end, error_update, error))


    def _build_database_hash_code_and_protos(self):
        dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='test',
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self._network.eval()
        hash_codes_list = []
        features_list = []
        targets_list = []

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                output = self._network(inputs)
                hash_codes_list.append(output["hash_code"])
                features_list.append(output["features"].cpu())
                targets_list.append(targets)

        all_hash_codes = torch.cat(hash_codes_list, dim=0)
        all_features = torch.cat(features_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        del hash_codes_list, features_list, targets_list

        all_targets_gpu = all_targets.to(self._device)
        if self._database_hashing_codes is None:
            self._database_hashing_codes = all_hash_codes
            self._database_codes_targets = all_targets_gpu
        else:
            self._database_hashing_codes = torch.cat([self._database_hashing_codes, all_hash_codes], dim=0)
            self._database_codes_targets = torch.cat([self._database_codes_targets, all_targets_gpu], dim=0)

        for class_idx in range(self._known_classes, self._total_classes):
            mask = (all_targets == class_idx)
            self._hash_protos.append(all_hash_codes[mask.to(self._device)].mean(dim=0))
            self._feature_class_protos.append(all_features[mask].mean(dim=0).to(self._device))

        del all_hash_codes, all_features, all_targets, all_targets_gpu


    def _get_projector(self, raw_matrix, explained_variance_ratio=0.98):
        if  explained_variance_ratio == 1:
            return torch.eye(raw_matrix.size(0), device=self._device), raw_matrix.size(0)
        eigenvals, eigenvecs = torch.linalg.eigh(raw_matrix)

        sorted_indices = torch.argsort(eigenvals, descending=True)
        eigenvals = eigenvals[sorted_indices]
        eigenvecs = eigenvecs[:, sorted_indices]
        
        explained_variance_ratio = explained_variance_ratio
        eigenvals_positive = torch.clamp(eigenvals, min=0)
        total_variance = eigenvals_positive.sum()
        cumulative_variance = torch.cumsum(eigenvals_positive, dim=0) / total_variance
        n_components = (cumulative_variance < explained_variance_ratio).sum().item() + 1
        n_components = min(n_components, eigenvecs.size(1)) 
        truncated_eigenvecs = eigenvecs[:, :n_components]
        
        P = truncated_eigenvecs @ truncated_eigenvecs.t()

        return P

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        criterion = Ori_Loss(self._hash_code_length)  # loss

        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            epoch_hash_loss = 0.0
            epoch_quant_loss = 0.0

            for i, (_, inputs, targets) in enumerate(train_loader):
                
                targets_onehot = target2onehot(targets, self._total_classes).float()
                inputs, targets_onehot, targets = inputs.to(self._device), targets_onehot.to(self._device), targets.float().to(self._device)
                
                output = self._network(inputs)
                hash_code = output["hash_code"]

                hash_loss, quant_loss = criterion(hash_code, targets_onehot)
                loss = hash_loss + self.args["gamma"] * quant_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_hash_loss += hash_loss.item()
                epoch_quant_loss += quant_loss.item()

            scheduler.step()
            
            if (epoch+1) % 10 == 0:
                query_codes, query_targets = self._generate_cur_hash_code_and_target(self._network, test_loader)
                database_codes, database_targets = self._generate_cur_hash_code_and_target(self._network, train_loader)
                test_map = self._compute_map(query_codes, query_targets, database_codes, database_targets)
                
                info = "Task {}, Epoch {}/{} => Loss: hash {:.5f}, quant {:.5f}, Current mAP {:.4f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    epoch_hash_loss / len(train_loader),
                    epoch_quant_loss / len(train_loader),
                    test_map,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss: hash {:.5f}, quant {:.5f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    epoch_hash_loss / len(train_loader),
                    epoch_quant_loss / len(train_loader)
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        criterion = Inc_Loss(self._hash_code_length, self._projector_loss)

        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            epoch_hash_loss_new = 0.0
            epoch_hash_loss_old = 0.0
            epoch_quant_loss = 0.0
            epoch_distill_loss = 0.0

            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = target2onehot(targets, self._total_classes).float()
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output_new = self._network(inputs)
                features_new, hash_codes_new = output_new["features"], output_new["hash_code"]

                with torch.no_grad():
                    features_old = self._network.backbone.forward_cumulative(inputs, adapter_index=self._cur_task-1)
                hash_codes_old = self._sample_hash_protos()

                hash_loss_new, hash_loss_old, quant_loss, distill_loss = criterion(hash_codes_new, targets, hash_codes_old, features_new, features_old)
                loss = hash_loss_new + hash_loss_old + self.args["gamma"] * quant_loss + self.args["mu"] * distill_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_hash_loss_new += hash_loss_new.item()
                epoch_hash_loss_old += hash_loss_old.item()
                epoch_quant_loss += quant_loss.item()
                epoch_distill_loss += distill_loss.item()

            scheduler.step()

            if (epoch+1) % 10 == 0:
                query_codes, query_targets = self._generate_cur_hash_code_and_target(self._network, test_loader)
                database_codes, database_targets = self._generate_cur_hash_code_and_target(self._network, train_loader)
                test_map = self._compute_map(query_codes, query_targets, database_codes, database_targets)
                
                info = "Task {}, Epoch {}/{} => Loss: hash new {:.5f}, hash old {:.5f}, quant {:.5f}, distill {:.5f}, Current mAP {:.4f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    epoch_hash_loss_new / len(train_loader),
                    epoch_hash_loss_old / len(train_loader),
                    epoch_quant_loss / len(train_loader),
                    epoch_distill_loss / len(train_loader),
                    test_map,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss: hash new {:.5f}, hash old {:.5f}, quants {:.5f}, distill {:.5f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["tuned_epoch"],
                    epoch_hash_loss_new / len(train_loader),
                    epoch_hash_loss_old / len(train_loader),
                    epoch_quant_loss / len(train_loader),
                    epoch_distill_loss / len(train_loader),
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _sample_hash_protos(self):

        hash_protos_tensor = torch.stack(self._hash_protos, dim=0)
        
        num_prototypes = hash_protos_tensor.size(0)
        if num_prototypes < self.batch_size:
            num_repeats = (self.batch_size + num_prototypes - 1) // num_prototypes
            repeated_prototypes = hash_protos_tensor.repeat(num_repeats, 1)
            omega = torch.randperm(repeated_prototypes.size(0))[: self.batch_size]
            epoch_hash_protos = repeated_prototypes[omega].sign()
        else:
            omega = torch.randperm(num_prototypes)[: self.batch_size]
            epoch_hash_protos = hash_protos_tensor[omega].sign()
        return epoch_hash_protos

    
class Ori_Loss(nn.Module):
    def __init__(self, code_length, eps=1e-6):
        super(Ori_Loss, self).__init__()
        self.code_length = code_length
        self.eps = eps

    def forward(self, Bn, labels):

        S = (labels @ labels.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
        r = S.sum() / ((1 - S).sum() + self.eps)
        S = S * (1 + r) - r
        dot_product = Bn @ Bn.t() / self.code_length
        hash_loss = ((S - dot_product) ** 2).mean()
        quant_loss = ((Bn.abs() - 1) ** 2).mean()

        return hash_loss, quant_loss



class Inc_Loss(nn.Module):
    def __init__(self, code_length, projector=None, eps=1e-6):
        super(Inc_Loss, self).__init__()
        self.code_length = code_length
        self.eps = eps
        self.projector = projector

    def forward(self, Bn, labels, Bo, features_new, features_old):

        S = (labels @ labels.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        r = S.sum() / ((1 - S).sum()+ self.eps)
        S = S * (1 + r) - r
        dot_product_new = Bn @ Bn.t() / self.code_length
        hash_loss_new = ((S - dot_product_new) ** 2).mean()

        So = -torch.ones(Bn.shape[0], Bo.shape[0], device=Bn.device)
        r = So.sum() / (1 - So).sum()
        So = So * (1 + r) - r
        dot_product_old = Bn @ Bo.t() / self.code_length
        hash_loss_old = ((So - dot_product_old) ** 2).mean()

        quant_loss = ((Bn.abs() - 1) ** 2).mean()
        cls_new_P = features_new @ self.projector.to(dtype=features_new.dtype, device=features_new.device)
        cls_old_P = features_old @ self.projector.to(dtype=features_old.dtype, device=features_old.device)
        distill_loss = ((cls_new_P - cls_old_P) ** 2).mean()

        return hash_loss_new, hash_loss_old, quant_loss, distill_loss
