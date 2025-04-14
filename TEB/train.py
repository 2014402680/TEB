import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_5fold, esm_embeddings_5fold
from model import GraphNet,GraphNet2,GraphNet3,GraphNet4,GraphNet5,GraphNet6,GraphNet7,GraphNet8,GraphNet9,GraphNet10,GraphNet11
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
import collections
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml


seed = 18
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)


def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)


def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)



args = parse_args()

with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

with open(args.configs_path2) as file:
    configs2 = yaml.safe_load(file)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# #data_list = esm_embeddings_5fold(args.configs_path)
# data_list2 = TEINet_embeddings_5fold(args.configs_path2)
# data_list2 = [data2.to(device) for data2 in data_list2]

#data_list = esm_embeddings_5fold(args.configs_path)
data_list = TEINet_embeddings_5fold(args.configs_path)
data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
test_data = data_list[1]


model = GraphNet(num_nodes=train_data.num_nodes,num_node_features=train_data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)


model3 = GraphNet3(num_node_features=train_data.num_node_features).to(device)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=args.lr, weight_decay=5e-4)

model4 = GraphNet4(num_node_features=train_data.num_node_features).to(device)
optimizer4 = torch.optim.Adam(model4.parameters(), lr=args.lr, weight_decay=5e-4)

model5 = GraphNet5(num_node_features=train_data.num_node_features).to(device)
optimizer5 = torch.optim.Adam(model5.parameters(), lr=args.lr, weight_decay=5e-4)

model6 = GraphNet6(num_node_features=train_data.num_node_features).to(device)
optimizer6 = torch.optim.Adam(model6.parameters(), lr=args.lr, weight_decay=5e-4)
#
# model7 = GraphNet7(num_node_features=train_data.num_node_features).to(device)
# optimizer7 = torch.optim.Adam(model7.parameters(), lr=args.lr, weight_decay=5e-4)
#
# model8 =GraphNet8(num_node_features=train_data.num_node_features).to(device)
# optimizer8 = torch.optim.Adam(model8.parameters(), lr=args.lr, weight_decay=5e-4)
#
# model9 = GraphNet9(num_node_features=train_data.num_node_features).to(device)
# optimizer9 = torch.optim.Adam(model9.parameters(), lr=args.lr, weight_decay=5e-4)
#
# model10 = GraphNet10(num_node_features=train_data.num_node_features).to(device)
# optimizer10 = torch.optim.Adam(model10.parameters(), lr=args.lr, weight_decay=5e-4)
#
# model11 = GraphNet11(num_node_features=train_data.num_node_features).to(device)
# optimizer11 = torch.optim.Adam(model11.parameters(), lr=args.lr, weight_decay=5e-4)

# train_data2 = data_list2[0]
# test_data2 = data_list2[1]
#
# model2 = GraphNet2(num_nodes=train_data2.num_nodes,num_node_features=train_data2.num_node_features).to(device)
# optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=5e-4)

margin = 4.0
epoch_decay = 0.0046
weight_decay = 0.006
aucm_optimizer = PESG(model.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)

#
# aucm_optimizer2 = PESG(model2.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)



aucm_optimizer3 = PESG(model3.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)
aucm_optimizer4 = PESG(model4.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)
aucm_optimizer5 = PESG(model5.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)
aucm_optimizer6 = PESG(model6.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)
# aucm_optimizer7 = PESG(model7.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)
# aucm_optimizer8 = PESG(model8.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)
# aucm_optimizer9 = PESG(model9.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)
# aucm_optimizer10 = PESG(model10.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)
# aucm_optimizer11 = PESG(model11.parameters(),
#                  loss_fn=AUCMLoss(),
#                  lr=args.lr,
#                  momentum=0.4,
#                  margin=margin,
#                  device=device,
#                  epoch_decay=epoch_decay,
#                  weight_decay=weight_decay)


num_epochs = args.epochs
best_valid_auc = 0
best_valid_aupr = 0
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    aucm_optimizer.zero_grad()

    out  = model(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)],device=device) * weight_factor * args.positive_weights
    bce_loss = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    #bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)

    aucm_module = AUCMLoss()
    aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
    total_loss = args.w_celoss * bce_loss + args.w_aucloss * aucm_loss.to(device)
    total_loss.backward()
    optimizer.step()
    aucm_optimizer.step()

    accuracy = compute_accuracy(preds, y_true)
    roc_auc = compute_auc(preds, y_true)
    aupr = compute_aupr(preds, y_true)

    # #model2
    # model2.train()
    # optimizer2.zero_grad()
    # aucm_optimizer2.zero_grad()
    #
    # out = model2(train_data2.x, train_data2.edge_index)
    # preds = out
    # y_true = train_data2.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss2 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module2 = AUCMLoss()
    # aucm_loss2 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss2 = args.w_celoss * bce_loss2 + args.w_aucloss * aucm_loss2.to(device)
    # total_loss2.backward()
    # optimizer2.step()
    # aucm_optimizer2.step()
    #
    # accuracy2 = compute_accuracy(preds, y_true)
    # roc_auc2 = compute_auc(preds, y_true)
    # aupr2 = compute_aupr(preds, y_true)

    model3.train()
    optimizer3.zero_grad()
    aucm_optimizer3.zero_grad()

    out = model3(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    bce_loss3 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)

    aucm_module3 = AUCMLoss()
    aucm_loss3 = aucm_module(torch.sigmoid(preds), y_true)
    total_loss3 = args.w_celoss * bce_loss3 + args.w_aucloss * aucm_loss3.to(device)
    total_loss3.backward()
    optimizer3.step()
    aucm_optimizer3.step()

    accuracy3 = compute_accuracy(preds, y_true)
    roc_auc3 = compute_auc(preds, y_true)
    aupr3 = compute_aupr(preds, y_true)

    model4.train()
    optimizer4.zero_grad()
    aucm_optimizer4.zero_grad()

    out = model4(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    bce_loss4 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)

    aucm_module4 = AUCMLoss()
    aucm_loss4 = aucm_module(torch.sigmoid(preds), y_true)
    total_loss4 = args.w_celoss * bce_loss4 + args.w_aucloss * aucm_loss4.to(device)
    total_loss4.backward()
    optimizer4.step()
    aucm_optimizer4.step()

    accuracy4 = compute_accuracy(preds, y_true)
    roc_auc4 = compute_auc(preds, y_true)
    aupr4 = compute_aupr(preds, y_true)

    model5.train()
    optimizer5.zero_grad()
    aucm_optimizer5.zero_grad()

    out = model5(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    bce_loss5 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)

    aucm_module5 = AUCMLoss()
    aucm_loss5 = aucm_module(torch.sigmoid(preds), y_true)
    total_loss5 = args.w_celoss * bce_loss5 + args.w_aucloss * aucm_loss5.to(device)
    total_loss5.backward()
    optimizer5.step()
    aucm_optimizer5.step()

    accuracy5 = compute_accuracy(preds, y_true)
    roc_auc5 = compute_auc(preds, y_true)
    aupr5 = compute_aupr(preds, y_true)

    model6.train()
    optimizer6.zero_grad()
    aucm_optimizer6.zero_grad()

    out = model6(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    bce_loss6 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)

    aucm_module6 = AUCMLoss()
    aucm_loss6 = aucm_module(torch.sigmoid(preds), y_true)
    total_loss6 = args.w_celoss * bce_loss6 + args.w_aucloss * aucm_loss6.to(device)
    total_loss6.backward()
    optimizer6.step()
    aucm_optimizer6.step()

    accuracy6 = compute_accuracy(preds, y_true)
    roc_auc6 = compute_auc(preds, y_true)
    aupr6 = compute_aupr(preds, y_true)

    # model7.train()
    # optimizer7.zero_grad()
    # aucm_optimizer7.zero_grad()

    # out = model7(train_data.x, train_data.edge_index)
    # preds = out
    # y_true = train_data.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss7 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module7 = AUCMLoss()
    # aucm_loss7 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss7 = args.w_celoss * bce_loss7 + args.w_aucloss * aucm_loss7.to(device)
    # total_loss7.backward()
    # optimizer7.step()
    # aucm_optimizer7.step()
    #
    # accuracy7 = compute_accuracy(preds, y_true)
    # roc_auc7 = compute_auc(preds, y_true)
    # aupr7 = compute_aupr(preds, y_true)
    #
    #
    # model8.train()
    # optimizer8.zero_grad()
    # aucm_optimizer8.zero_grad()
    #
    # out = model8(train_data.x, train_data.edge_index)
    # preds = out
    # y_true = train_data.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss8 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module8 = AUCMLoss()
    # aucm_loss8 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss8 = args.w_celoss * bce_loss8 + args.w_aucloss * aucm_loss8.to(device)
    # total_loss8.backward()
    # optimizer8.step()
    # aucm_optimizer8.step()
    #
    # accuracy8 = compute_accuracy(preds, y_true)
    # roc_auc8 = compute_auc(preds, y_true)
    # aupr8 = compute_aupr(preds, y_true)
    #
    # model9.train()
    # optimizer9.zero_grad()
    # aucm_optimizer9.zero_grad()
    #
    # out = model9(train_data.x, train_data.edge_index)
    # preds = out
    # y_true = train_data.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss9 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module9 = AUCMLoss()
    # aucm_loss9 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss9 = args.w_celoss * bce_loss9 + args.w_aucloss * aucm_loss9.to(device)
    # total_loss9.backward()
    # optimizer9.step()
    # aucm_optimizer9.step()
    #
    # accuracy9 = compute_accuracy(preds, y_true)
    # roc_auc9 = compute_auc(preds, y_true)
    # aupr9 = compute_aupr(preds, y_true)
    #
    # model10.train()
    # optimizer10.zero_grad()
    # aucm_optimizer10.zero_grad()
    #
    # out = model10(train_data.x, train_data.edge_index)
    # preds = out
    # y_true = train_data.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss10 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module10 = AUCMLoss()
    # aucm_loss10 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss10 = args.w_celoss * bce_loss10 + args.w_aucloss * aucm_loss10.to(device)
    # total_loss10.backward()
    # optimizer10.step()
    # aucm_optimizer10.step()
    #
    # accuracy10 = compute_accuracy(preds, y_true)
    # roc_auc10 = compute_auc(preds, y_true)
    # aupr10 = compute_aupr(preds, y_true)
    #
    # model11.train()
    # optimizer11.zero_grad()
    # aucm_optimizer11.zero_grad()
    #
    # out = model11(train_data.x, train_data.edge_index)
    # preds = out
    # y_true = train_data.y.to(device)
    #
    # num_positive_samples = (y_true == 1).sum()
    # num_negative_samples = (y_true == 0).sum()
    # weight_factor = num_negative_samples.float() / num_positive_samples.float()
    # pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor * args.positive_weights
    # bce_loss11 = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    # # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true)
    #
    # aucm_module11 = AUCMLoss()
    # aucm_loss11 = aucm_module(torch.sigmoid(preds), y_true)
    # total_loss11 = args.w_celoss * bce_loss11 + args.w_aucloss * aucm_loss11.to(device)
    # total_loss11.backward()
    # optimizer11.step()
    # aucm_optimizer11.step()
    #
    # accuracy11 = compute_accuracy(preds, y_true)
    # roc_auc11 = compute_auc(preds, y_true)
    # aupr11 = compute_aupr(preds, y_true)


    best_weights = [0.14, 0.14]
    # Validation part
    model.eval()
    # model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    # model7.eval()
    # model8.eval()
    # model9.eval()
    # model10.eval()
    # model11.eval()

    with torch.no_grad():
        out_valid = model(test_data.x, test_data.edge_index)
        out_sum = 0
        out_valid *= 0.14

        # out_valid_2 = model2(test_data2.x, test_data2.edge_index)
        # out_valid_2 *= 0.14

        out_valid_3 = model3(test_data.x, test_data.edge_index)
        out_valid_3 *= 0.14

        out_valid_4 = model4(test_data.x, test_data.edge_index)
        out_valid_4 *= 0.14

        out_valid_5 = model5(test_data.x, test_data.edge_index)
        out_valid_5 *= 0.14

        out_valid_6 = model6(test_data.x, test_data.edge_index)
        out_valid_6 *= 0.14

        # out_valid_7 = model7(test_data.x, test_data.edge_index)
        # out_valid_7 *= 0.14
        #
        # out_valid_8 = model8(test_data.x, test_data.edge_index)
        # out_valid_8 *= 0.14
        #
        # out_valid_9 = model9(test_data.x, test_data.edge_index)
        # out_valid_9 *= 0.14
        #
        # out_valid_10 = model10(test_data.x, test_data.edge_index)
        # out_valid_10 *= 0.14
        #
        # out_valid_11 = model11(test_data.x, test_data.edge_index)
        # out_valid_11 *= 0.14

        out_sum += out_valid + out_valid_3 + out_valid_4 + out_valid_5 + out_valid_6

        out_sum /= 0.7 + 0.2

        #out_sum /= 0.28 + 0.2

        out_valid=out_sum
        preds_valid = out_valid
        print(out_valid)
        y_true_valid = test_data.y.to(device)


        valid_acc = compute_accuracy(preds_valid, y_true_valid)
        roc_auc_valid = compute_auc(preds_valid, y_true_valid)
        valid_aupr = compute_aupr(preds_valid, y_true_valid)

        if roc_auc_valid > best_valid_auc:
            best_valid_auc = roc_auc_valid
            best_valid_aupr= valid_aupr
            torch.save(model.state_dict(), configs['save_model'])
            #torch.save(model2.state_dict(), configs2['save_model'])
    
    print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Train AUC: {:.4f}, Train APUR: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(epoch+1, num_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))

print(" Test AUC: {:.4f}, Test AUPR: {:.4f}".format(best_valid_auc, best_valid_aupr))

# Load the best model
best_model = GraphNet(num_nodes=train_data.num_nodes,num_node_features=test_data.num_node_features).to(device)
best_model.load_state_dict(torch.load(configs['save_model']))
#
# best_model2 = GraphNet2(num_nodes=train_data2.num_nodes,num_node_features=test_data2.num_node_features).to(device)
# best_model2.load_state_dict(torch.load(configs2['save_model']))
#
# # Evaluate on test test_data
# best_model.eval()
# best_model2.eval()
# with torch.no_grad():
#     out_test = best_model(test_data.x, test_data.edge_index)
#     out_sum = 0
#     best_weights = [0.14, 0.14]
#     out_test *= 0.14
#
#     out_valid_2 = best_model2(test_data2.x, test_data2.edge_index)
#     out_valid_2 *= 0.14
#
#     out_sum += out_test + out_valid_2
#
#     out_sum /= 0.28 + 0.2
#
#     preds_test = out_sum
#     y_true_test = test_data.y.to(device)
#
#     test_acc = compute_accuracy(preds_test, y_true_test)
#     roc_auc_test = compute_auc(preds_test, y_true_test)
#     test_aupr = compute_aupr(preds_test, y_true_test)
#
#     # save results
#     probabilities = torch.sigmoid(preds_test)
#     binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
#     df = pd.DataFrame({
#         'prediction': binary_predictions,
#         'label': y_true_test.detach().cpu().numpy().astype(int)
#     })
#     df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)
#
#
# print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(test_acc, roc_auc_test, test_aupr))