import numpy as np
import torch
import torch.nn as nn
import re
import functools
from collections import OrderedDict

import util.util as util
import  os
pretrained_path = "/home/yunzb/zt/age-ode/Lifespan_Age_Transformation_Synthesis/train_result_weight/latest_net_G.pth"
pretrained_dict = torch.load(pretrained_path)
keys=list(pretrained_dict.keys())
update_pretrained_dict = {k: v for k, v in pretrained_dict.items() }
print(update_pretrained_dict)