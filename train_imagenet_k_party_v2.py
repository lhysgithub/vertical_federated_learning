import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from models.manual_k_party import Manual_A, Manual_B
from dataset import get_train_dataset, get_val_dataset
from torchviz import make_dot
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.optim import *

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', required=True, help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=0, help='num of workers')
parser.add_argument('--epochs', type=int, default=350, help='num of training epochs')
parser.add_argument('--layers', type=int, default=18, help='total number of layers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=350, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--k', type=int, required=True, help='num of client')
parser.add_argument('--ratio', type=float, default=1.0, help='portion of train samples')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer')

args = parser.parse_args()

args.name = 'experiments/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*/*.py') + glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
exp_name = f'{args.k}_party_{args.optimizer}_lr{args.learning_rate}_bz{args.batch_size}_epochs_{args.epochs}_ratio_' \
           f'{args.ratio}.txt'
fh = logging.FileHandler(os.path.join(args.name, exp_name))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# config
train_model = "cloak"
save_model_epoch = 50
dlg_epoch = 50
start_train_epoch = 0


def main():
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    model_A = Manual_A(num_classes=10, layers=args.layers, k=args.k, in_channel=1, width=0.5)
    if args.k == 1:
        model_list = [model_A]
    elif args.k == 2:
        model_B = Manual_B(in_channel=2, layers=18, width=0.5)
        model_list = [model_A, model_B]
    else:
        assert ValueError
    model_list = [model.to(device) for model in model_list]

    # model_list = load_model_list(model_list, 50, train_model)


    for i in range(args.k):
        logging.info("model_{} param size = {}MB".format(i + 1, utils.count_parameters_in_MB(model_list[i])))

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer_list = [torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                          weight_decay=args.weight_decay) for model in model_list]
    else:
        optimizer_list = [torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
                          for model in model_list]
    train_data = get_train_dataset(args.data)
    valid_data = get_val_dataset(args.data)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    if args.learning_rate == 0.025:
        scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            for optimizer in optimizer_list]
    else:
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]

    best_acc_top1 = 0

    Intermediate = 64
    stds = torch.ones(Intermediate, requires_grad=True)
    means = torch.zeros(Intermediate, requires_grad=True)
    optimizer_noise = torch.optim.Adam([means, stds], args.learning_rate)
    lambdaa = 1

    for epoch in range(start_train_epoch, args.epochs):
        lr = scheduler_list[0].get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train_with_noise(train_queue, model_list, criterion, optimizer_list, epoch,
                                                means, stds, lambdaa, optimizer_noise)
        [scheduler_list[i].step() for i in range(len(scheduler_list))]
        logging.info('train_acc %f', train_acc)

        valid_acc_top1, valid_obj = infer(valid_queue, model_list, criterion, epoch)
        logging.info('valid_acc_top1 %f', valid_acc_top1)

        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
        logging.info('best_acc_top1 %f', best_acc_top1)

        if epoch != 0 and epoch != start_train_epoch and epoch % save_model_epoch == 0:
            save_model_list(model_list, epoch, train_model)


def save_model_list(model_list, epoch, train_model):
    for i, model in enumerate(model_list):
        utils.save(model, f"saved_models/{train_model}_{epoch}_{i}.pt.tar")


def load_model_list(model_list, epoch, train_model):
    new_model_list = []
    for i, model in enumerate(model_list):
        utils.load(model, f"saved_models/{train_model}_{epoch}_{i}.pt.tar")
        new_model_list.append(model)
    return new_model_list


def train_with_noise(train_queue, model_list, criterion, optimizer_list, epoch, means, stds, lambdaa, optimizer_noise):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    cur_step = epoch * len(train_queue)

    model_list = [model.train() for model in model_list]
    k = len(model_list)

    for step, (trn_x, trn_y) in enumerate(train_queue):
        # cloak
        noise = torch.randn(means.shape)
        std = (torch.tanh(stds*-2) + torch.ones_like(stds)) / 2 # 初始值为0.5
        noise = noise * std + means
        noise = noise.repeat((trn_x.shape[0], 1)).to(device)
        # cloak

        trn_X = torch.split(trn_x, [1, 2], dim=1)  # 64，299，299，3 -> 64,1,224,224 + 64,2,224,224
        trn_X = [x.float().to(device) for x in trn_X]
        target = trn_y.view(-1).long().to(device)
        n = target.size(0)
        [optimizer_list[i].zero_grad() for i in range(k)]
        U_B_list = None
        U_B_clone_list = None
        if k > 1:
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]  #
            U_B_clone_list = [U_B.detach().clone() + noise for U_B in U_B_list]
            U_B_clone_list = [torch.autograd.Variable(U_B, requires_grad=True) for U_B in U_B_clone_list]
        logits = model_list[0](trn_X[0], U_B_clone_list)
        loss = criterion(logits, target)

        if k > 1:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_clone_list]
            model_B_weights_gradients_list = [
                torch.autograd.grad(U_B_list[i], model_list[i + 1].parameters(), grad_outputs=U_B_gradients_list[i],
                                    retain_graph=True) for i in range(len(U_B_gradients_list))]
            for i in range(len(model_B_weights_gradients_list)):
                for w, g in zip(model_list[i + 1].parameters(), model_B_weights_gradients_list[i]): # todo: change the gradient to parameters
                    w.grad = g.detach()
                nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                optimizer_list[i + 1].step()

        loss.backward(retain_graph=True)

        # cloak
        grad = torch.autograd.grad(loss, U_B_clone_list[0], retain_graph=True)[0]
        grad1 = grad.detach()
        if train_model == "cloak":
            grad2 = torch.nn.functional.normalize(grad1, p=2.0, dim=1, eps=1e-12, out=None)
            grad4 = torch.abs(grad2)
            loss2 = torch.mean(grad4.to("cpu") * torch.log(std)) - lambdaa * torch.mean(torch.log(std))
            # debug
            optimizer_noise.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer_noise.step()
            # debug
        # cloak

        # dlg
        # if epoch != 0 and epoch % dlg_epoch == 0 and step == 0:
        if epoch % dlg_epoch == 0 and step == 0:
            dummy_data, dummy_label = inverting_gradients_single(model_list, grad1, trn_X, target, U_B_clone_list[0],model_B_weights_gradients_list[0])
            # dummy_data, dummy_label = deep_leakage_from_gradients_single(model_list, grad1, trn_X, target, U_B_clone_list[0],model_B_weights_gradients_list[0])  # todo: test DLG
            original_img = trn_x[0]
            original_one_chanel_img = trn_X[1][0]
            revers_img = dummy_data[0]
            save_image(original_img, f"original_img_{epoch}.png")
            save_image(original_one_chanel_img, f"original_one_chanel_img_{epoch}.png")
            save_image(revers_img, f"revers_img_{epoch}.png")
        # dlg

        nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
        optimizer_list[0].step()

        prec1 = utils.accuracy(logits, target)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss1 {losses.avg:.3f} "
                "Prec ({top1.avg:.1f}%) Loss2 {losses2:.3f} Noise@(means,stds.mean,stds.max) ({means:.1f},{stds:.1f},{stdsmax:.1f})".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=objs, top1=top1,
                    losses2=torch.mean(loss2), means=torch.mean(means), stds=torch.mean(std), stdsmax=torch.max(std)))
        cur_step += 1
    return top1.avg, objs.avg


def getL2NormFromTuples(tuple1,tuple2):
    sum1 = torch.tensor(0).to(device)
    for i, j in zip(tuple1, tuple2):
        sum1 = sum1 + (torch.abs(i - j)).sum()
    return sum1


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def inverting_gradients_single(model_list, origin_grad_batch, origin_data_batch, origin_label_batch, origin_embedding_batch, origin_model_grad):
    origin_grad = origin_grad_batch[0].unsqueeze(0)
    origin_data0 = origin_data_batch[0][0].unsqueeze(0)
    origin_data1 = origin_data_batch[1][0].unsqueeze(0)
    origin_label = origin_label_batch[0].unsqueeze(0)
    origin_embedding = origin_embedding_batch[0].unsqueeze(0)
    dummy_data = torch.randn(origin_data1.size(), requires_grad=True)
    dummy_label = torch.randn(F.one_hot(origin_label, num_classes=10).size(), requires_grad=True)
    # optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.1)
    # optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.01)
    optimizer = torch.optim.Adam([dummy_data], lr=0.1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    epoch_up = 24000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_up // 2.667, epoch_up // 1.6, epoch_up // 1.142], gamma=0.1)
    # optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    criterion = cross_entropy_for_onehot

    dummy_data_old = dummy_data.clone()
    # origin_data1_max = origin_data1.max()
    # origin_data1_min = origin_data1.min()
    model_list[0].eval()
    model_list[1].eval()
    model_grad_copy = []
    tvloss = TVLoss()
    for i in origin_model_grad:
        model_grad_copy.append(i.detach().clone())

    for iters in range(epoch_up):
        # dummy_data_max = dummy_data.max()
        # dummy_data_min = dummy_data.min()
        # dummy_data_change = dummy_data - dummy_data_old
        # dummy_data_change_max = dummy_data_change.max()
        # dummy_data_change_min = dummy_data_change.min()
        optimizer.zero_grad()
        dummy_pred = model_list[1](dummy_data.to(device))
        logits = model_list[0](origin_data0, [dummy_pred])
        # _, dummy_targe = F.softmax(dummy_label, dim=-1).max(dim=-1) # 此处产生了梯度断层
        # dummy_targe = F.softmax(dummy_label, dim=-1)
        # dummy_loss = criterion(logits, dummy_targe.to(device))
        # dummy_loss = criterion(logits, dummy_targe.long().to(device))
        # dummy_loss = criterion(logits, dummy_label.to(device))
        dummy_loss = criterion(logits, origin_label.to(device))
        dummy_grad = torch.autograd.grad(dummy_loss, dummy_pred, create_graph=True)
        dummy_model_grad = torch.autograd.grad(dummy_loss, model_list[1].parameters(), create_graph=True)

        # dummy_grad_diff = sum(((dummy_grad - origin_grad) ** 2).sum() \
        #                 for dummy_g, origin_g in zip(dummy_grad, origin_grad))

        # original DLG
        # dummy_grad_diff = ((dummy_grad[0] - origin_grad) ** 2).sum()

        # dummy_model_grad_diff = getL2NormFromTuples(dummy_model_grad, origin_model_grad)
        # dummy_model_grad_diff = sum(((dummy_g - origin_g) ** 2).sum() for dummy_g, origin_g in zip(dummy_model_grad, model_grad_copy))

        dummy_pred_diff = ((origin_embedding - dummy_pred) ** 2).sum()

        dummy_grad_diff = (torch.abs(dummy_grad[0] - origin_grad)).sum() * 1

        # dummy_grad_cos_diff = 1-torch.cosine_similarity(dummy_grad[0], origin_grad)
        dummy_grad_cos_diff = reconstruction_costs(dummy_grad[0], origin_grad)

        # dummy_tv_loss = tvloss(dummy_data)
        dummy_tv_loss = total_variation(dummy_data)

        dummy_data_diff = ((origin_data1 - dummy_data.to(device)) ** 2).sum()

        dummy_label_diff = ((dummy_label.to(device) - F.one_hot(origin_label, num_classes=10)) ** 2).sum()

        # loss = dummy_grad_cos_diff + dummy_tv_loss + dummy_grad_diff
        # loss = dummy_grad_cos_diff + dummy_tv_loss
        # loss = dummy_grad_diff + dummy_pred_diff + dummy_model_grad_diff
        # loss = dummy_model_grad_diff
        # loss = dummy_grad_cos_diff + dummy_tv_loss + dummy_grad_diff + dummy_pred_diff + dummy_model_grad_diff
        # loss = dummy_grad_cos_diff + dummy_tv_loss+dummy_data_diff
        loss = dummy_grad_cos_diff + 0.1*dummy_tv_loss

        if iters % 100 == 0:
            logging.info(f"DLG [{iters}/{epoch_up}] "
                         f"loss {loss}\t"
                         # f"dummy_grad_diff {dummy_grad_diff}\t"
                         # f"dummy_model_grad_diff {dummy_model_grad_diff}\t"
                         f"dummy_grad_cos_diff {dummy_grad_cos_diff.data}\t"
                         f"dummy_tv_loss {dummy_tv_loss}\t"
                         f"dummy_pred_diff {dummy_pred_diff}\t"
                         f"dummy_data_diff {dummy_data_diff}\t"
                         f"dummy_label_diff {dummy_label_diff}\t"
                         )
        # grad = dummy_data.grad
        loss.backward()
        # grad1 = dummy_data.grad
        # grad1_max = grad1.max()
        # grad1_min = grad1.min()
        # dummy_data_old = dummy_data.clone()

        optimizer.step()
        scheduler.step()

    model_list[0].train()
    model_list[1].train()
    return dummy_data, dummy_label


def inverting_gradients_batch(model_list, origin_grad_batch, origin_data_batch, origin_label_batch, origin_embedding_batch, origin_model_grad):
    origin_grad = origin_grad_batch[0].unsqueeze(0)
    origin_data0 = origin_data_batch[0][0].unsqueeze(0)
    origin_data1 = origin_data_batch[1][0].unsqueeze(0)
    origin_label = origin_label_batch[0].unsqueeze(0)
    origin_embedding = origin_embedding_batch[0].unsqueeze(0)
    dummy_data = torch.randn(origin_data1.size(), requires_grad=True)
    dummy_label = torch.randn(F.one_hot(origin_label, num_classes=10).size(), requires_grad=True)
    # optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD([dummy_data, dummy_label], lr=0.1)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    # optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    criterion = cross_entropy_for_onehot
    epoch_up = 24000
    dummy_data_old = dummy_data.clone()
    # origin_data1_max = origin_data1.max()
    # origin_data1_min = origin_data1.min()
    model_list[0].eval()
    model_list[1].eval()
    model_grad_copy = []
    tvloss = TVLoss()
    for i in origin_model_grad:
        model_grad_copy.append(i.detach().clone())

    for iters in range(epoch_up):
        optimizer.zero_grad()
        dummy_pred = model_list[1](dummy_data.to(device))
        logits = model_list[0](origin_data0, [dummy_pred])
        dummy_loss = criterion(logits, dummy_label.to(device))
        dummy_grad = torch.autograd.grad(dummy_loss, dummy_pred, create_graph=True)

        dummy_pred_diff = ((origin_embedding - dummy_pred) ** 2).sum()

        dummy_grad_diff = (torch.abs(dummy_grad[0] - origin_grad)).sum() * 1

        dummy_grad_cos_diff = 1-torch.cosine_similarity(dummy_grad[0], origin_grad)

        dummy_tv_loss = tvloss(dummy_data)

        # loss = dummy_grad_cos_diff + dummy_tv_loss + dummy_grad_diff
        loss = dummy_grad_cos_diff + dummy_tv_loss
        # loss = dummy_grad_diff + dummy_pred_diff + dummy_model_grad_diff
        # loss = dummy_model_grad_diff
        # loss = dummy_grad_cos_diff + dummy_tv_loss + dummy_grad_diff + dummy_pred_diff + dummy_model_grad_diff

        dummy_data_diff = ((origin_data1 - dummy_data.to(device)) ** 2).sum()

        dummy_label_diff = ((dummy_label.to(device) - F.one_hot(origin_label, num_classes=10)) ** 2).sum()

        if iters % 100 == 0:
            logging.info(f"DLG [{iters}/{epoch_up}] "
                         f"loss {loss}\t"
                         f"dummy_grad_diff {dummy_grad_diff}\t"
                         # f"dummy_model_grad_diff {dummy_model_grad_diff}\t"
                         f"dummy_grad_cos_diff {dummy_grad_cos_diff.data}\t"
                         f"dummy_tv_loss {dummy_tv_loss}\t"
                         f"dummy_pred_diff {dummy_pred_diff}\t"
                         f"dummy_data_diff {dummy_data_diff}\t"
                         f"dummy_label_diff {dummy_label_diff}\t"
                         )
        loss.backward()

        optimizer.step()
        scheduler.step()

    model_list[0].train()
    model_list[1].train()
    return dummy_data, dummy_label


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def infer(valid_queue, model_list, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model_list = [model.eval() for model in model_list]
    k = len(model_list)
    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_queue):
            val_X = torch.split(val_X, [1, 2], dim=1)
            val_X = [x.float().to(device) for x in val_X]
            target = val_y.view(-1).long().to(device)
            n = target.size(0)
            U_B_list = None
            if k > 1:
                U_B_list = [model_list[i](val_X[i]) for i in range(1, len(model_list))]
            logits = model_list[0](val_X[0], U_B_list)
            loss = criterion(logits, target)
            prec1= utils.accuracy(logits, target)
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1f}%)".format(
                        epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                        top1=top1))
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
