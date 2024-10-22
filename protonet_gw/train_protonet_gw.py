import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
from models.protonet_gw import ProtoNet
from models.post_hnn import Post_hnn
import optimizers
from hyptorch.pmath import dist_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import (
    pprint,
    set_gpu,
    ensure_path,
    Averager,
    Timer,
    count_acc,
    compute_confidence_interval,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet10",
        choices=[
            "convnet",
            "resnet10",
            "resnet12",
            "resnet18",
        ],
    )
    parser.add_argument("--post_model", type=str, default="Post_HNN", choices=["Post_HNN"])
    parser.add_argument("--type", type=str, default="post", choices=["post"])
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--validation_way", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--post_lr", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--post_step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--post_gamma", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument(
        "--dataset", type=str, default="CUB", choices=["MiniImageNet", "CUB"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--hyperbolic", action="store_true", default=True)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--num-layers_post", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=int, default=1)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not-riemannian", action="store_true")
    parser.add_argument(
        "--manifold", type=str, default="PoincareBall", choices=["Euclidean", "Hyperboloid", "PoincareBall"]
    )

    parser.add_argument(
        "--post_optimizer", type=str, default="RiemannianAdam", choices=["Adam", "RiemannianAdam"]
    )
    args = parser.parse_args()
    pprint(vars(args))
    args.riemannian = not args.not_riemannian

    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
        args.device = 'cuda:' + str(args.cuda)
        print(args.device)

    if args.save_path is None:
        save_path1 = "-".join([args.dataset, "ProtoNet"])
        save_path2 = "_".join(
            [   
                str(args.beta),
                str(args.type),
                str(args.max_epoch),
                str(args.shot),
                str(args.query),
                str(args.way),
                str(args.validation_way),
                str(args.step_size),
                str(args.gamma),
                str(args.lr),
                str(args.post_lr),
                str(args.temperature),
                str(args.hyperbolic),
                str(args.dim),
                str(args.h_dim),
                str(args.c)[:5],
                str(args.num_layers_post),
                str(args.post_lr),
                str(args.model),
            ]
        )
        args.save_path = save_path1 + "_" + save_path2
        ensure_path(args.save_path)
    else:
        ensure_path(args.save_path)

    if args.dataset == "MiniImageNet":
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == "CUB":
        from dataloader.cub import CUB as Dataset
        args.feat_dim = 3*84*84
    else:
        raise ValueError("Non-supported Dataset.")

    trainset = Dataset("train", args)
    train_sampler = CategoriesSampler(
        trainset.label, 100, args.way, args.shot + args.query
    )
    train_loader = DataLoader(
        dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True
    )

    valset = Dataset("val", args)
    val_sampler = CategoriesSampler(
        valset.label, 500, args.validation_way, args.shot + args.query
    )
    val_loader = DataLoader(
        dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True
    )


    model = ProtoNet(args)
    print(model)

    post_optimizer = getattr(optimizers, args.post_optimizer)(params=model.gwd.parameters(), lr=args.post_lr)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)

    if args.lr_decay:
        post_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            post_optimizer, step_size=args.post_step_size, gamma=args.post_gamma
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    model_dict = model.state_dict()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)["params"]
        pretrained_dict = {"encoder." + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)


    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    def save_model(name):  
        torch.save(
            dict(params=model.state_dict()), osp.join(args.save_path, name + ".pth")
        )

    trlog = {}
    trlog["args"] = vars(args)
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["train_acc"] = []
    trlog["val_acc"] = []
    trlog["max_acc"] = 0.0
    trlog["max_acc_epoch"] = 0


    timer = Timer()
    global_count = 0
    writer = SummaryWriter(comment=args.save_path)


    for epoch in range(1, args.max_epoch + 1):
        if args.lr_decay:
            lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]

            logits, gwd = model(data_shot, data_query)
            loss = F.cross_entropy(logits, label) + args.beta * gwd

            acc = count_acc(logits, label)
            writer.add_scalar("data/loss", float(loss), global_count)
            writer.add_scalar("data/acc", float(acc), global_count)
            print(
                "epoch {}, train {}/{}, loss={:.4f} acc={:.4f}".format(
                    epoch, i, len(train_loader), loss.item(), acc
                )
            )

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            post_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            post_optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        label = torch.arange(args.validation_way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        print(
            "best epoch {}, best val acc={:.4f}".format(
                trlog["max_acc_epoch"], trlog["max_acc"]
            )
        )
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.validation_way
                data_shot, data_query = data[:p], data[p:]

                logits, gwd = model(data_shot, data_query)

                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar("data/val_loss", float(vl), epoch)
        writer.add_scalar("data/val_acc", float(va), epoch)
        print("epoch {}, val, loss={:.4f} acc={:.4f}".format(epoch, vl, va))

        if va > trlog["max_acc"]:
            trlog["max_acc"] = va
            trlog["max_acc_epoch"] = epoch
            save_model("max_acc")

        trlog["train_loss"].append(tl)
        trlog["train_acc"].append(ta)
        trlog["val_loss"].append(vl)
        trlog["val_acc"].append(va)

        torch.save(trlog, osp.join(args.save_path, "trlog"))

        save_model("epoch-last")

        print(
            "ETA:{}/{}".format(timer.measure(), timer.measure(epoch / args.max_epoch))
        )
    writer.close()

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, "trlog"))
    test_set = Dataset("test", args)
    sampler = CategoriesSampler(
        test_set.label, 10000, args.validation_way, args.shot + args.query
    )
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.load_state_dict(
        torch.load(osp.join(args.save_path, "max_acc" + ".pth"))["params"]
    )
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.validation_way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    for i, batch in enumerate(loader, 1):
        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in batch]
        else:
            data = batch[0]
        k = args.validation_way * args.shot
        data_shot, data_query = data[:k], data[k:]

        logits, gwd = model(data_shot, data_query)
        acc = count_acc(logits, label)
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc
        print("batch {}: {:.2f}({:.2f})".format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    print(
        "Val Best Acc {:.4f}, Test Acc {:.4f}".format(trlog["max_acc"], ave_acc.item())
    )
    print("Test Acc {:.4f} + {:.4f}".format(m, pm))
