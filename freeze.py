import time
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from pdb import set_trace as st
# from  ipdb import set_trace
pthfile="/home/yunzb/zt/age-ode/Lifespan_Age_Transformation_Synthesis/checkpoints/males_model/latest_net_D.pth"
net = torch.load(pthfile)
dict_new = net.state_dict().copy()