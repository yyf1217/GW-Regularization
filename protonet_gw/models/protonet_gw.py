import os
import sys

import torch
import manifolds

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from networks.convnet import ConvNet
from networks.ResNet import resnet18, resnet10
from networks.ResNet12 import Res12
import models.encoders as encoders
from models.post_hnn import Post_hnn


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.model
        post_model_name = args.post_model

        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        

        if model_name == "convnet":
            self.encoder = ConvNet(z_dim=args.dim)
        elif model_name == "resnet10":
            self.encoder = resnet10(remove_linear=True)
        elif model_name == "resnet12":
            self.encoder = Res12()
        elif model_name == "resnet18":
            self.encoder = resnet18(remove_linear=True)
        else:
            raise ValueError("Model not found")

        if post_model_name == "Post_HNN":
            self.manifold = getattr(manifolds, self.manifold_name)()        
            self.gwd = getattr(encoders, args.post_model)(self.c, args)
        


    def forward(self, data_shot, data_query):
        if self.args.hyperbolic:
            e_embedding_shot = self.encoder(data_shot)
            h_embedding_shot = self.gwd.encode(e_embedding_shot)

            if self.training:
                proto = h_embedding_shot.reshape(self.args.shot, self.args.way, -1)
            else:
                proto = h_embedding_shot.reshape(self.args.shot, self.args.validation_way, -1)

            proto = poincare_mean(proto, dim=0, c=self.c)

            e_embedding_query = self.encoder(data_query)
            h_embedding_query = self.gwd.encode(e_embedding_query)


            # Calculating distances in Euclidean space
            e_embedding = torch.cat((e_embedding_shot, e_embedding_query), dim=0)
            x = e_embedding
            diff_matrix = x.unsqueeze(1) - x.unsqueeze(0)
            xy = torch.norm(diff_matrix**2, p=2, dim=2)

            
            
            # Calculating distances in Hyperbolic space
            h_embedding = torch.cat((h_embedding_shot, h_embedding_query), dim=0) 
            Tx = h_embedding           
            Txy = dist_matrix(Tx, Tx, c=self.c)


            # Calculating GW distances
            cal_choose = 0    # GW distances can be calculated using different types
            if cal_choose = 0:
                Axy = xy
                ATxy = Txy
            elif cal_choose = 1:
                xy_mean = torch.mean(xy)
                Axy = (xy - xy_mean) / xy_mean
                Txy_mean = torch.mean(Txy)
                ATxy = (Txy - Txy_mean) / Txy_mean
            elif cal_choose = 2:
                Axy = 1.0/(1.0+xy)
                ATxy = 1.0/(1.0+Txy)
            elif cal_choose = 3:
                Axy = torch.log(1.0 + xy)
                ATxy = torch.log(1.0 + Txy)
            
            gwd = torch.mean((Axy - ATxy)**2)

            
            logits = (
                -dist_matrix(h_embedding_query, proto, c=self.c) / self.args.temperature
            )

        else:
            proto = self.encoder(data_shot)
            if self.training:
                proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            else:
                proto = proto.reshape(
                    self.args.shot, self.args.validation_way, -1
                ).mean(dim=0)

            logits = (
                euclidean_metric(self.encoder(data_query), proto)
                / self.args.temperature
            )
        return logits, gwd
