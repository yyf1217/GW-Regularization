import os
import sys

import torch
import manifolds

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
import models.encoders as encoders


class Post_hnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.pre_c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        

        
        self.encoder = getattr(encoders, args.pre_model)(self.c, args)



    def forward(self, x_data):
        if self.args.hyperbolic:
            H_embed = self.encoder.encode(x_data)


        return H_embed
