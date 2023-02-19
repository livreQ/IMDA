#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue May 10 22:48:42 2022
# File Name: src/main/run_IMDA.py
# Description:
"""


from copy import deepcopy
import os
from statistics import mode
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch


from src.algo.IMDA import IMDABase
from src.model.model import MLPNet, MLPNet_digits,ConvNet
from utils.solver import Convex, BBSL, NLLSL
from utils.data_loader import load_numpy_data, data_loader, multi_data_loader, shift_trainset
import utils.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from numpy import linalg as LA

class IMDA_MLP(IMDABase):
    def __init__(self, args, configs):
        """
        IMDA with MLP
        """
        super().__init__(args, configs)

        fea_configs = {
            "input_dim": configs["input_dim"],
            "hidden_layers": configs["hidden_layers"][:-1],
            "output_dim": configs["hidden_layers"][-1],
            "drop_rate": configs["drop_rate"],
            "process_final": True,
        }
        self.feature_net = MLPNet(fea_configs)
        self.un_feature_net = MLPNet(fea_configs)
        self.class_net = nn.Linear(configs["hidden_layers"][-1], configs["num_classes"])
        self.su_W_net = nn.Linear(configs["hidden_layers"][-1], configs["num_classes"])
        self.un_W_net = nn.Linear(configs["hidden_layers"][-1], configs["num_classes"])


class IMDA_Conv_Digits(IMDABase):
    def __init__(self, args, configs):
        """
        IMDA with convolution feature extractor
        """

        super().__init__(args, configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {
            "input_dim": configs["input_dim"],
            "hidden_layers": configs["cls_fc_layers"],
            "output_dim": configs["num_classes"],
            "drop_rate": configs["drop_rate"],
            "process_final": False,
        }
        self.class_net = MLPNet_digits(cls_configs)
        self.su_W_net = MLPNet_digits(cls_configs)
        self.un_W_net = MLPNet_digits(cls_configs)



def build_label_shift_MT_datasets(args, num_datasets, tar_dom_idx, train_insts, train_labels):
    if args.transfer_type == "sup":#supervised
        percent = args.percent
    elif args.transfer_type == "unsup": # unsupervised
        percent = 1
    else:
        percent = args.percent
    if args.name == "amazon":
        # for Amazon
        src_shift_labels = [0]
        src_drop_ratios = [args.drop]
    elif args.name == "digits":
        # for digits
        src_shift_labels = [5, 6, 7, 8, 9]
        src_drop_ratios = [args.drop]*5
    # Build source instances
    source_insts = []
    source_labels = []

    for j in range(num_datasets):
        if j != tar_dom_idx:
            train_x_temps, train_y_temps = shift_trainset(
                train_insts[j].astype(np.float32),
                train_labels[j].astype(np.int64),
                src_shift_labels,
                src_drop_ratios,
            )
            source_insts.append(train_x_temps)
            source_labels.append(train_y_temps)
    # Build target instances
    # construct a random drop 90% of the data (save a limited target lablled data)
    target_x_temps = train_insts[tar_dom_idx].astype(np.float32)
    target_y_temps = train_labels[tar_dom_idx].astype(np.int64)
    label_idx = np.arange(len(target_y_temps))
    np.random.shuffle(label_idx)
    num_drop = int(np.ceil(len(target_y_temps) * percent))
    dropped_idx = label_idx[:num_drop]
    # tar_test_idx = label_idx[:num_drop]
    # the dropped 90% data are treated as test set (since they are unseen)
    tar_test_insts = np.take(target_x_temps, dropped_idx, axis=0)
    tar_test_labels = np.take(target_y_temps, dropped_idx, axis=0)
    target_insts = np.delete(target_x_temps, dropped_idx, axis=0)
    target_labels = np.delete(target_y_temps, dropped_idx, axis=0)

    return (source_insts, source_labels, target_insts, target_labels, tar_test_insts, tar_test_labels)


def get_label_distribution(num_src_domains, num_src_classes, source_labels, target_labels):
    # Compute ground truth source/ target label distribution (normalized)
    src_label_P = np.zeros([num_src_domains, num_src_classes])
    tar_label_P = np.zeros([num_src_classes])
    for tsk in range(num_src_domains):
        for j in range(num_src_classes):
            src_label_P[tsk, j] = np.count_nonzero(source_labels[tsk] == j)
        src_label_P[tsk, :] = src_label_P[tsk, :] / len(source_labels[tsk])

    for j in range(num_src_classes):
        tar_label_P[j] = np.count_nonzero(target_labels == j)
    tar_label_P = tar_label_P / len(target_labels)
    return src_label_P, tar_label_P




def learn(args):
    """
        run multi-source transfer algorithm
    """
    ###################### configuration #################
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    torch.autograd.set_detect_anomaly(True)

    exp_flags = "lr_{:g}_tau_{:g}_epsilon_{:g}_gp_{:g}_W1_sup_ceof_{:g}_W1_unsup_coef_{:g}_W1_discri_coef1_{:g}_l2_scale_{:g}_epoch_{:d}_W1_discri_coef2_{:g}_seed_{:d}_drop_{:g}_percent_{:g}"\
        .format(args.lr, args.tau, args.epsilon, args.gp_coef, args.W1_sup_coef, args.W1_unsup_coef,
         args.W1_discri_coef1, args.l2_scale, args.epoch, args.W1_discri_coef2, args.seed, args.drop, args.percent)
    result_path = os.path.join(args.result_path, args.name, args.transfer_type, args.alpha_reg_type, exp_flags)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logger = utils.get_logger(os.path.join(result_path, "log_{}.log".format(exp_flags)))
    logger.info("Hyperparameter setting = %s" % args)

    # Set random number seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #################### Load the datasets ####################
    print(torch.__version__)
    time_start = time.time()

    data_names, train_insts, train_labels, test_insts, test_labels, data_configs = load_numpy_data(args.name, args.data_path, logger)
    # number of source classes,
    num_src_classes = data_configs["num_classes"]
    # number of sources + target
    num_datasets = len(data_names)

    logger.info("Time used to process the %s = %g seconds." % (args.name, time.time() - time_start))
    logger.info("-" * 100)
    # choose one dataset as target and the rest as sources
    test_results = {}
    np_test_results = np.zeros(num_datasets)


    #################### Model ####################
    num_src_domains = data_configs["num_src_domains"]
    logger.info("Model setting = %s." % data_configs)

    #################### Train ####################

    lambda_list = np.zeros([num_datasets, num_src_domains, args.epoch])

    for tar_dom_idx, tar_dom_name in enumerate(data_names):
        # collect source data names from full data names except for target data name
        src_data_names = [name for name in data_names if name != tar_dom_name]
        # display sources v.s. target
        logger.info("*" * 100)
        logger.info("*  Source domains: [{}], target domain: [{}] ".format("/".join(src_data_names), tar_dom_name))
        logger.info("*" * 100)

        #################### Build label shift multi-source transfer data sets ################


        source_insts, source_labels, target_insts, target_labels, \
            tar_test_insts, tar_test_labels = build_label_shift_MT_datasets(args, num_datasets, tar_dom_idx, train_insts, train_labels)
        m_s = []
        for src_label in source_labels:
            m_s.append(src_label.shape[0])
        m_s = np.asarray(m_s)
        m_t = max(target_insts.shape[0], 1)
        m_t_prime = max(tar_test_insts.shape[0], 1)
        logger.info("#samples in source domains for train m_s:= {}".format(m_s))
        logger.info("#labeled samples in target domain for train m_t:= {}".format(m_t))
        logger.info("#unlabeled samples in target domain for train m_t_prime:= {}".format(m_t_prime))
        logger.info("#samples in target domain for test = {}".format(len(tar_test_labels)))

        #### calculate source target label distribution ####
        #src_label_P, tar_label_P = get_label_distribution(num_src_domains, num_src_classes, source_labels, target_labels)

        # ##################### Model ######################
        if args.name in ["amazon", "office_home"]:  # MLP
            model = IMDA_MLP(args, data_configs).to(device)
        elif args.name == "digits":  # ConvNet
            model = IMDA_Conv_Digits(args, data_configs).to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # Training phase
        model.train()
        time_start = time.time()
        # defining lambda and alpha (global)
        model.alpha = torch.from_numpy(np.ones([num_src_domains]) / num_src_domains).to(device)
        L2_regularization = 1

        global_step = 0
        U_g, V_g = 0, 0
        G_norm_bound = 0
        ##########  start training epoch ####################
        for epoch_idx in range(args.epoch):
            model.train()
            running_loss = 0.0
            R_T_epoch, R_S_alpha_epoch, W1_sup_epoch, W1_unsup_epoch, domain_loss_epoch = 0,0,0,0,0
            cvx_losses = np.zeros(num_src_domains)
            train_loader = multi_data_loader(source_insts, source_labels, batch_size)

            for batch_idx, (xs, ys) in enumerate(
                tqdm(train_loader, desc="Epoch {}...".format(epoch_idx + 1))):
                global_step += 1

                for j in range(num_src_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)

                if args.transfer_type == "sup":
                    ridx = np.random.choice(target_insts.shape[0], batch_size)
                    xt = target_insts[ridx, :]
                    xt = torch.tensor(xt, requires_grad=False).to(device)
                    yt = target_labels[ridx]
                    yt = torch.tensor(yt, requires_grad=False).to(device)
                    xt_prime = None
                elif args.transfer_type == "unsup":
                    xt = None
                    yt = None
                    uidx = np.random.choice(tar_test_insts.shape[0], batch_size)
                    xt_prime = tar_test_insts[uidx, :]
                    xt_prime = torch.tensor(xt_prime, requires_grad=False).to(device)
                else:
                    ridx = np.random.choice(target_insts.shape[0], batch_size)
                    xt = target_insts[ridx, :]
                    xt = torch.tensor(xt, requires_grad=False).to(device)
                    yt = target_labels[ridx]
                    yt = torch.tensor(yt, requires_grad=False).to(device)
                    uidx = np.random.choice(tar_test_insts.shape[0], batch_size)
                    xt_prime = tar_test_insts[uidx, :]
                    xt_prime = torch.tensor(xt_prime, requires_grad=False).to(device)

                train_loss, convex_loss, R_T, R_S_alpha, W1_sup, W1_unsup, domain_loss, u_g_norm, v_g_norm = model(xs, ys, xt, yt, xt_prime)
                pure_train_loss = train_loss.item()
                optimizer.zero_grad()
                if args.add_gnorm_penalty:
                    train_loss += args.gp_coef * (u_g_norm + v_g_norm)
                    # u_grad = torch.autograd.grad(train_loss, model.feature_net.parameters(),retain_graph=True)
                    # v_grad = torch.autograd.grad(train_loss, model.class_net.parameters(),retain_graph=True)
                    # # #print(u_grad.shape, v_grad.shape)
                    # u_g_norm = model.cal_grad_norm(u_grad)#torch.mean(torch.sqrt(torch.sum(u_grad ** 2, dim=1) + 1e-10)**2)
                    # v_g_norm = model.cal_grad_norm(v_grad)#torch.mean(torch.sqrt(torch.sum(v_grad ** 2, dim=1) + 1e-10)**2)
                train_loss.backward()
                ############## SGLD ###########
                if args.sgld:
                    noise_var = np.sqrt(2*args.lr/args.temp)
                    #g_u, g_v = [], []
                    for p in model.feature_net.parameters():
                        # g_u.append(deepcopy(p.grad))
                        p.grad += torch.normal(torch.zeros_like(p), noise_var)
                    for p in model.class_net.parameters():
                        # g_v.append(deepcopy(p.grad))
                        p.grad += torch.normal(torch.zeros_like(p), noise_var)

                    U_g += u_g_norm.cpu().numpy() * args.lr #* args.temp / 2
                    V_g += v_g_norm.cpu().numpy() * args.lr #* args.temp / 2
                    tmp_alpha = model.alpha.cpu().numpy()
                    #print(type(tmp_alpha))
                    G1 = args.tau * args.epsilon * np.sqrt(U_g * (LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2 + 1.0/m_t))
                    G2 = args.tau * np.sqrt((V_g + U_g)*(((1 - args.epsilon)**2) / m_t + args.epsilon ** 2 * ((LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2))))
                    G3 = (1 - args.tau) * np.sqrt((U_g + V_g) * (LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2 + 1.0/m_t))
                    G_norm_bound = G1 + G2 + G3
                    # args.tau * args.epsilon * np.sqrt(U_g * (LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2 + 1.0/m_t))\
                    #      + args.tau * np.sqrt((V_g + U_g)((1 - args.epsilon) **2 /m_t + args.epsilon ** 2 * ((LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2))))\
                    #         + (1 - args.tau) * np.sqrt((U_g + V_g) * (LA.norm(tmp_alpha * np.sqrt(1.0/m_s)) ** 2 + 1.0/m_t_prime))
                    #print(noise_var, U_g, V_g, u_g_norm, v_g_norm)
                optimizer.step()
                with torch.no_grad():
                    # convert to L2 optimization mode
                    loss_np = convex_loss.cpu().numpy()
                    cvx_losses += loss_np
                running_loss += pure_train_loss#train_loss.item()
                R_T_epoch += R_T.item()
                R_S_alpha_epoch += R_S_alpha.item()
                W1_sup_epoch += W1_sup.item()
                W1_unsup_epoch += W1_unsup.item()
                domain_loss_epoch += domain_loss.item()
            cvx_losses /= batch_idx + 1
            ######################## update alpha #########################
            if args.transfer_type == "unsup": # unsupervised
                if args.name == "digits":
                    START_EPOCH = 5
                elif args.name == "amazon":
                    START_EPOCH = 1
                # update lambda
                if args.name == "digits":
                    L2_regularization = np.sum(cvx_losses)
                if epoch_idx > START_EPOCH and epoch_idx % 1 == 0:
                    if args.alpha_reg_type == "g_norm":
                        L2_regularization = args.l2_scale * ((1 - args.tau + 2 * args.tau * args.epsilon)* np.sqrt(U_g) + (args.tau*args.epsilon)*np.sqrt(V_g))
                        task_alpha = Convex(cvx_losses, L2_regularization, m_s)
                        L2_regularization /= np.sqrt(np.mean(m_s))
                    else:
                        task_alpha = Convex(cvx_losses, L2_regularization)
                    model.alpha = 0.8 * model.alpha + 0.2 * torch.from_numpy(task_alpha).to(device)
            else:
                if epoch_idx > 1 and epoch_idx % 3 == 0:
                    if args.alpha_reg_type == "g_norm":
                        L2_regularization = args.l2_scale * ((1 - args.tau + 2 * args.tau * args.epsilon)* np.sqrt(U_g) + (args.tau*args.epsilon)*np.sqrt(V_g))
                        task_alpha = Convex(cvx_losses, L2_regularization, m_s)
                        L2_regularization /= np.sqrt(np.mean(m_s))
                    else:
                        L2_regularization = np.max(cvx_losses)# change to gradient norm
                    #L2_regularization = (1 - args.tau + 2 * args.tau * args.epsilon)* np.sqrt(U_g) + (args.tau*args.epsilon)*np.sqrt(V_g)
                        task_alpha = Convex(cvx_losses, L2_regularization)
                    model.alpha = 0.8 * model.alpha + 0.2 * torch.from_numpy(task_alpha).to(device)
                # else:
                #     logger.info("Epoch[{}/{}], no updates for lambda!".format(epoch_idx + 1, args.epoch))

            ##################### display ####################
            lambdas_in_str = [" {}:{:.6f} ".format(dom_name, model.alpha[idx].cpu()) for idx, dom_name in enumerate(src_data_names)]
            logger.info("Epoch[{}/{}], Lambda=[{}]".format(epoch_idx + 1, args.epoch, ",".join(lambdas_in_str)))
            lambda_list[tar_dom_idx, :, epoch_idx] = model.alpha.cpu()
            logger.info("Epoch[{}/{}], running_loss = {:.4f}".format(epoch_idx + 1, args.epoch, running_loss/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], R_S_alpha_epoch = {:.4f}".format(epoch_idx + 1, args.epoch, R_S_alpha_epoch/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], R_T_epoch = {:.4f}".format(epoch_idx + 1, args.epoch, R_T_epoch/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], W1_sup_epoch = {:.4f}".format(epoch_idx + 1, args.epoch, W1_sup_epoch/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], W1_unsup_epoch = {:.4f}".format(epoch_idx + 1, args.epoch, W1_unsup_epoch/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], domain_loss_epoch = {:.4f}".format(epoch_idx + 1, args.epoch, domain_loss_epoch/(batch_idx + 1)))
            logger.info("Epoch[{}/{}], L2_regularization = {:.4f}".format(epoch_idx + 1, args.epoch, L2_regularization))
            logger.info("Epoch[{}/{}], Gnorm_bound = {:.4f}".format(epoch_idx + 1, args.epoch, G_norm_bound))
            logger.info("Epoch[{}/{}], RHS_bound = {:.4f}".format(epoch_idx + 1, args.epoch, running_loss/(batch_idx + 1) + G_norm_bound))
            logger.info("Finish training in {:.6g} seconds".format(time.time() - time_start))
            model.eval()

            ############### Test (use another hold-out target) ##############
            n_test = 1
            if args.transfer_type == 0:#supervised
                test_loader = data_loader(tar_test_insts, tar_test_labels, batch_size=1000, shuffle=False)
                n_test = tar_test_labels.shape[0]
            else:
                test_loader = data_loader(test_insts[tar_dom_idx], test_labels[tar_dom_idx], batch_size=1000, shuffle=False)
                n_test = test_insts[tar_dom_idx].shape[0]
            test_acc = 0.0
            test_loss = 0.0
            cnt = 0
            for xt, yt in test_loader:
                xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
                yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
                preds_labels = torch.argmax(model.inference(xt), 1)
                test_loss += F.nll_loss(model.inference(xt), yt)
                test_acc += torch.sum(preds_labels == yt).item()
                cnt += 1
            test_acc /= n_test
            test_loss /= cnt
            logger.info("Epoch[{}/{}], test loss on [{}] = {:.6g}".format(epoch_idx + 1, args.epoch, tar_dom_name, test_loss))
            logger.info("Epoch[{}/{}], test accuracy on [{}] = {:.6g}".format(epoch_idx + 1, args.epoch, tar_dom_name, test_acc))
            test_results[tar_dom_name] = test_acc
            np_test_results[tar_dom_idx] = test_acc
        logger.info("All test accuracies: ")
        logger.info(test_results)

        # Save results to files
        with open(os.path.join(result_path, "test_{}.txt".format(exp_flags)), "w") as test_file:
            avg_test_acc = 0
            for tar_dom_name, test_acc in test_results.items():
                test_file.write("{} = {:.6g}\n".format(tar_dom_name, test_acc))
                avg_test_acc += test_acc
            avg_test_acc /= len(test_results.items())
            test_file.write("{} = {:.6g}\n".format("Average acc", avg_test_acc))

        logger.info("Finish {}_{}".format(exp_flags, tar_dom_name))
        logger.info("*" * 100)
    logger.info("All finished!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset: [amazon|digits].", type=str, choices=["amazon", "digits"], default="amazon")
    parser.add_argument("--result_path", help="Where to save results.", type=str, default="./results")
    parser.add_argument("--data_path", help="Where to find the data.", type=str, default="./datasets")
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.4)
    parser.add_argument("--drop", help="drop rate.", type=float, default=0.5)
    parser.add_argument("--percent", help="Label percentage.", type=float, default=0.9)
    parser.add_argument("--add_gnorm_penalty", help="add gnorm penalty", type=int, default=1)
    parser.add_argument("--alpha_reg_type", help="task weight alpha cvx solver reg type", type=str, choices=["g_norm", "wadn"], default="g_norm")
    parser.add_argument("--transfer_type", help="supervised, unsupervised, semi-supervised", type=str, choices=["sup","unsup",'semi'], default="unsup")
    parser.add_argument("--tau",help="Hyperparameter for semisupervised transfer weights, tau on supervised , 1-tau on unsupervised data",type=float, default=0)
    parser.add_argument("--epsilon", help="weight for target loss for supervised transfer", type=float, default=1.0)
    parser.add_argument("--gp", help="Coefficent of Wasserstein gradient penality.", type=float, default=1.0)
    parser.add_argument("--gp_coef", help="Coefficent of gradient penality loss(gp_coef).", type=float, default=1.0)
    parser.add_argument('--temp', type=int, help='sgld temparature', default=100000000)
    parser.add_argument('--sgld', type=int, help='use sgld or not, 0-sgd', default=1)
    parser.add_argument('--l2_scale', type=float, help='l2_scale for cvx penalty', default=1)
    parser.add_argument('--W1_sup_coef', type=float, help='penaty for supervised W1 distance', default=0.01)
    parser.add_argument('--W1_unsup_coef', type=float, help='penaty for unsupervised W1 distance', default=1)
    parser.add_argument('--W1_discri_coef1', type=float, help='penaty for unsupervised W1 distance term 1', default=0.1)
    parser.add_argument('--W1_discri_coef2', type=float, help='penaty for unsupervised W1 distance term 2', default=1)
    parser.add_argument("--epoch", help="Number of training epochs.", type=int, default=50)
    parser.add_argument("--batch_size", help="Batch size during training.", type=int, default=20)
    parser.add_argument("--cuda", help="Which cuda device to use.", type=int, default=0)
    parser.add_argument("--seed", help="Random seed.", type=int, default=0)
    args = parser.parse_args()
    learn(args)
