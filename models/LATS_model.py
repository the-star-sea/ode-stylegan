### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import re
import functools
from collections import OrderedDict
from .base_model import BaseModel
import util.util as util
import os
# from .networks import *
from . import networks
from pdb import set_trace as st
import os
from pathlib import Path
import torch.nn.functional as F


class LATS(BaseModel):  # Lifetime Age Transformation Synthesis
    def name(self):
        return 'LATS'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # if opt.resize_or_crop != 'none': # when training at full res this causes OOM
        torch.backends.cudnn.benchmark = True

        # determine mode of operation [train, test, deploy, traverse (latent interpolation)]
        self.isTrain = opt.isTrain
        self.traverse = (not self.isTrain) and opt.traverse

        # mode to generate Fig. 15 in the paper
        self.compare_to_trained_outputs = (not self.isTrain) and opt.compare_to_trained_outputs
        if self.compare_to_trained_outputs:
            self.compare_to_trained_class = opt.compare_to_trained_class
            self.trained_class_jump = opt.trained_class_jump

        self.deploy = (not self.isTrain) and opt.deploy
        if not self.isTrain and opt.random_seed != -1:
            torch.manual_seed(opt.random_seed)
            torch.cuda.manual_seed_all(opt.random_seed)
            np.random.seed(opt.random_seed)

        # network architecture parameters
        self.nb = opt.batchSize
        self.size = opt.fineSize
        self.ngf = opt.ngf
        self.ngf_global = self.ngf

        self.numClasses = opt.numClasses
        self.age_classes = [2, 3, 2, 4, 9, 9]
        self.use_moving_avg = not opt.no_moving_avg

        self.no_cond_noise = opt.no_cond_noise
        style_dim = opt.gen_dim_per_style * self.numClasses  # 300 是可以手动更改的
        self.duplicate = opt.gen_dim_per_style

        self.cond_length = style_dim

        # self.active_classes_mapping = opt.active_classes_mapping

        if not self.isTrain:
            self.debug_mode = opt.debug_mode
        else:
            self.debug_mode = False
        # #### age estimate model
        # age_estimate.init_model()
        ##### define networks
        # Generators
        self.netG = self.parallelize(networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                                       id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids,
                                                       padding_type='reflect', style_dim=style_dim,
                                                       init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                                       decoder_norm=opt.decoder_norm, activation=opt.activation,
                                                       adaptive_blocks=opt.n_adaptive_blocks,
                                                       normalize_mlp=opt.normalize_mlp,
                                                       modulated_conv=opt.use_modulated_conv))
        self.age_estimate_model = self.parallelize(networks.def_age_estimate(gpu_ids=self.gpu_ids))
        if self.isTrain and self.use_moving_avg:
            self.g_running = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsample,
                                               id_enc_norm=opt.id_enc_norm, gpu_ids=self.gpu_ids,
                                               padding_type='reflect', style_dim=style_dim,
                                               init_type='kaiming', conv_weight_norm=opt.conv_weight_norm,
                                               decoder_norm=opt.decoder_norm, activation=opt.activation,
                                               adaptive_blocks=opt.n_adaptive_blocks, normalize_mlp=opt.normalize_mlp,
                                               modulated_conv=opt.use_modulated_conv)
            self.g_running.train(False)
            self.requires_grad(self.g_running, flag=False)
            self.accumulate(self.g_running, self.netG, decay=0)

        # Discriminator network
        if self.isTrain:
            self.netD = self.parallelize(networks.define_D(opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                                           numClasses=self.numClasses, gpu_ids=self.gpu_ids,
                                                           init_type='kaiming'))

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        def load_total_work(model, pretrained_path, label='run'):
            model_dict = model.state_dict()
            for k, v in model_dict.items():
                print("original:", k)
            pretrained_dict_path = pretrained_path
            # for pretrained_dict_path in items:
            pretrained_dict = torch.load(pretrained_dict_path)

            if label == 'G' or label == 'D':
                keys = list(pretrained_dict.keys())
                for j in keys:
                    pretrained_dict.update({"module." + j: pretrained_dict.pop(j)})

            for k, v in pretrained_dict.items():
                #     # set_trace()
                print("pretrain", k)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            update_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            #
            # print('can be loaded', update_pretrained_dict.keys())
            model.load_state_dict(update_pretrained_dict, strict=False)
            # from sets import Set
            not_initialized = set()
            for k, v in model.state_dict().items():
                if k not in update_pretrained_dict or v.size() != update_pretrained_dict[k].size():
                    # set_trace()
                    not_initialized.add(k)
            print('unload', sorted(not_initialized))

        pretrained_path = "/home/yunzb/zt/age-ode/Lifespan_Age_Transformation_Synthesis/pretrain_model/males_model"
        if self.isTrain:
            load_total_work(self.netG, os.path.join(pretrained_path, "latest_net_G.pth"), 'G')
            load_total_work(self.netD, os.path.join(pretrained_path, "latest_net_D.pth"), 'D')

            load_total_work(self.g_running, os.path.join(pretrained_path, "latest_net_g_running.pth"))
            ## load age model
            resume_path = Path(__file__).resolve().parent.joinpath("age_estimation_pytorch", "misc",
                                                                   "epoch044_0.02343_3.9984.pth")
            checkpoint1 = torch.load(resume_path)
            self.age_estimate_model.load_state_dict(checkpoint1['state_dict'], strict=False)
        else:
            pretrained_path = "/home/yunzb/zt/age-ode/Lifespan_Age_Transformation_Synthesis/checkpoints/males_model"
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            # load_total_work(self.netG, os.path.join(pretrained_path, "latest_net_G.pth"), 'G')
        # # load networks
        # if (not self.isTrain) or opt.continue_train or opt.load_pretrain:
        #     pretrained_path = '' if (not self.isTrain) or (self.isTrain and opt.continue_train) else opt.load_pretrain
        #     if self.isTrain:
        #         self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
        #         self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
        #         if self.use_moving_avg:
        #             self.load_network(self.g_running, 'g_running', opt.which_epoch, pretrained_path)
        #     elif self.use_moving_avg:
        #         self.load_network(self.netG, 'g_running', opt.which_epoch, pretrained_path)
        #     else:
        #         self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            # define loss functions
            self.criterionGAN = self.parallelize(networks.SelectiveClassesNonSatGANLoss())
            self.R1_reg = networks.R1_reg()
            self.age_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.identity_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.criterionCycle = self.parallelize(networks.FeatureConsistency())  # torch.nn.L1Loss()
            self.criterionRec = self.parallelize(networks.FeatureConsistency())  # torch.nn.L1Loss()
            self.latent_reconst_criterion = self.parallelize(networks.FeatureConsistency())
            self.age_bias_loss = self.parallelize(networks.FeatureConsistency())  # age loss
            # initialize optimizers
            self.old_lr = opt.lr

            # set optimizer G
            paramsG = []
            params_dict_G = dict(self.netG.named_parameters())
            # set the MLP learning rate to 0.01 or the global learning rate
            for key, value in params_dict_G.items():
                decay_cond = ('decoder.mlp' in key)
                if opt.decay_adain_affine_layers:
                    decay_cond = decay_cond or ('class_std' in key) or ('class_mean' in key)
                if decay_cond:
                    paramsG += [{'params': [value], 'lr': opt.lr * 0.01, 'mult': 0.01}]
                else:
                    paramsG += [{'params': [value], 'lr': opt.lr}]
            # TODO
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, params_dict_G.values()), lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))

            # set optimizer D
            paramsD = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, paramsD), lr=opt.lr,
                                                betas=(opt.beta1, opt.beta2))

            # self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def parallelize(self, model):
        # parallelize a network
        if self.isTrain and len(self.gpu_ids) > 0:
            return networks._CustomDataParallel(model)
        else:
            return model

    def requires_grad(self, model, flag=True):
        # freeze network weights
        for p in model.parameters():
            p.requires_grad = flag

    def accumulate(self, model1, model2, decay=0.999):
        # implements exponential moving average
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        model1_parallel = isinstance(model1, nn.DataParallel)
        model2_parallel = isinstance(model2, nn.DataParallel)

        for k in params1.keys():
            if model2_parallel and not model1_parallel:
                k2 = 'module.' + k
            elif model1_parallel and not model2_parallel:
                k2 = re.sub('module.', '', k)
            else:
                k2 = k
            params1[k].data.mul_(decay).add_(1 - decay, params2[k2].data)

    def set_inputs(self, data, mode='train'):
        # set input data to feed to the network
        if mode == 'train':
            real_A = data['A']
            real_B = data['B']

            self.class_A = data['A_class']
            self.class_B = data['B_class']

            self.reals = torch.cat((real_A, real_B), 0)

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()

        else:
            inputs = data['Imgs']
            if inputs.dim() > 4:
                inputs = inputs.squeeze(0)

            self.class_A = data['Classes']
            if self.class_A.dim() > 1:
                self.class_A = self.class_A.squeeze(0)

            if torch.is_tensor(data['Valid']):
                self.valid = data['Valid'].bool()
            else:
                self.valid = torch.ones(1, dtype=torch.bool)

            if self.valid.dim() > 1:
                self.valid = self.valid.squeeze(0)

            if isinstance(data['Paths'][0], tuple):
                self.image_paths = [path[0] for path in data['Paths']]
            else:
                self.image_paths = data['Paths']

            self.isEmpty = False if any(self.valid) else True
            if not self.isEmpty:
                available_idx = torch.arange(len(self.class_A))
                select_idx = torch.masked_select(available_idx, self.valid).long()
                inputs = torch.index_select(inputs, 0, select_idx)

                self.class_A = torch.index_select(self.class_A, 0, select_idx)
                self.image_paths = [val for i, val in enumerate(self.image_paths) if self.valid[i] == 1]

            self.reals = inputs

            if len(self.gpu_ids) > 0:
                self.reals = self.reals.cuda()

    def get_conditions(self, mode='train'):
        # set conditional inputs to the network
        if mode == 'train':
            nb = self.reals.shape[0] // 2
        elif self.traverse or self.deploy:
            if self.traverse and self.compare_to_trained_outputs:
                nb = 2
            else:
                nb = self.numClasses
        else:
            nb = self.numValid

        # tex condition mapping
        condG_A_gen = self.Tensor(nb, self.cond_length)
        condG_B_gen = self.Tensor(nb, self.cond_length)
        condG_A_orig = self.Tensor(nb, self.cond_length)
        condG_B_orig = self.Tensor(nb, self.cond_length)

        if self.no_cond_noise:
            noise_sigma = 0
        else:
            noise_sigma = 0.2

        for i in range(nb):
            # 对应的类位置+1
            condG_A_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_gen[i, self.class_B[i] * self.duplicate:(self.class_B[i] + 1) * self.duplicate] += 1  # 这里填充了B class
            if not (self.traverse or self.deploy):
                condG_B_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_gen[i, self.class_A[i] * self.duplicate:(self.class_A[i] + 1) * self.duplicate] += 1

                condG_A_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_A_orig[i, self.class_A[i] * self.duplicate:(self.class_A[i] + 1) * self.duplicate] += 1

                condG_B_orig[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
                condG_B_orig[i, self.class_B[i] * self.duplicate:(self.class_B[i] + 1) * self.duplicate] += 1

        if mode == 'train':
            self.gen_conditions = torch.cat((condG_A_gen, condG_B_gen), 0)  # torch.cat((self.class_B, self.class_A), 0)
            # if the results are not good this might be the issue!!!! uncomment and update code respectively
            self.cyc_conditions = torch.cat((condG_B_gen, condG_A_gen), 0)
            self.orig_conditions = torch.cat((condG_A_orig, condG_B_orig), 0)
        else:
            self.gen_conditions = condG_A_gen  # self.class_B
            if not (self.traverse or self.deploy):
                # if the results are not good this might be the issue!!!! uncomment and update code respectively
                self.cyc_conditions = condG_B_gen  # self.class_A
                self.orig_conditions = condG_A_orig

    def ca_age_bias(self, images, A, B, age_bias_loss):
        start_age = None
        classes = torch.cat((A, B), 0)
        for i, ob in enumerate(images):
            new_images = ob.clone()
            new_images = new_images.resize_(new_images.shape[0], new_images.shape[1], 224, 224)
            outputs = F.softmax(self.age_estimate_model(new_images), dim=-1)
            ages = torch.arange(0, 101, device='cuda')
            if i == 0:
                start_age = (outputs * ages).sum(axis=-1)
            else:
                predicted_ages = (outputs * ages).sum(axis=-1)
                predicted_ages -= start_age
                for j in range(predicted_ages.shape[0]):
                    predicted_ages[j] /= self.age_classes[classes[j]]
                delta = torch.ones_like(predicted_ages) * i / len(images)
                print(predicted_ages)
                age_bias_loss += self.age_bias_loss(predicted_ages, delta)

            # for j in range(ob.shape[0] // 2):
            #     c = self.age_classes[A[j]]
            #     e_age = age_estimate.estimate_age(ob[j])
            #     if i == 0:
            #         age_list.append(e_age)
            #     else:
            #         delta_age = (e_age - age_list[j]) / c
            #         t = torch.tensor(i / len(images), device='cuda')
            #         print(type(delta_age))
            #         print(type(t))
            #         age_bias_loss += self.age_bias_loss(delta_age, t)
            # for j in range(ob.shape[0] // 2, ob.shape[0]):
            #     c = self.age_classes[B[j - ob.shape[0] // 2]]
            #     e_age = age_estimate.estimate_age(ob[j])
            #     if i == 0:
            #         age_list.append(e_age)
            #     else:
            #         delta_age = (e_age - age_list[j]) / c
            #         t = torch.tensor(i / len(images), device='cuda')
            #         age_bias_loss += self.age_bias_loss(delta_age, t)
        return age_bias_loss

    def update_G(self, infer=False):
        # Generator optimization setp
        self.optimizer_G.zero_grad()
        self.get_conditions()

        ############### multi GPU ###############
        rec_images, gen_images, cyc_images, orig_id_features, \
        orig_age_features, fake_id_features, fake_age_features, l1, l2, l3, l4 = \
            self.netG(self.reals, self.gen_conditions, self.cyc_conditions, self.orig_conditions)
        disc_outs = []
        # discriminator pass
        with torch.no_grad():
            for ob in gen_images:
                disc_outs.append(self.netD(ob))
        # age_estimate_loss
        age_bias_loss = self.age_bias_loss(self.reals, self.reals)
        self.ca_age_bias(rec_images, self.class_A, self.class_B, age_bias_loss)*self.opt.lambda_age_bias
        self.ca_age_bias(gen_images, self.class_B, self.class_A, age_bias_loss)*self.opt.lambda_age_bias
        self.ca_age_bias(cyc_images, self.class_A, self.class_B, age_bias_loss)*self.opt.lambda_age_bias
        # real_img_bias = age_estimate.estimate_age(self.reals)

        # self-reconstruction loss
        if self.opt.lambda_rec > 0:
            loss_G_Rec = self.criterionRec(rec_images[19], self.reals) * self.opt.lambda_rec
        else:
            loss_G_Rec = torch.zeros(1).cuda()

        # cycle loss
        if self.opt.lambda_cyc > 0:
            loss_G_Cycle = self.criterionCycle(cyc_images[19], self.reals) * self.opt.lambda_cyc
        else:
            loss_G_Cycle = torch.zeros(1).cuda()

        # identity feature loss
        loss_G_identity_reconst = self.identity_reconst_criterion(orig_id_features,
                                                                  orig_id_features) * self.opt.lambda_id
        for ob in fake_id_features:
            loss_G_identity_reconst += self.identity_reconst_criterion(ob, orig_id_features) * self.opt.lambda_id
        print("identity loss: ", loss_G_identity_reconst.mean(), self.opt.lambda_id)
        # age feature loss
        loss_G_age_reconst = self.age_reconst_criterion(self.gen_conditions, self.gen_conditions) * self.opt.lambda_age
        for ob in fake_age_features:
            loss_G_age_reconst += self.age_reconst_criterion(ob, self.gen_conditions) * self.opt.lambda_age
        print("target age  loss: ", loss_G_age_reconst.mean(), self.opt.lambda_age)
        # orig age feature loss
        loss_G_age_reconst += self.age_reconst_criterion(orig_age_features, self.orig_conditions) * self.opt.lambda_age

        # adversarial loss
        target_classes = torch.cat((self.class_B, self.class_A), 0)
        loss_G_GAN = self.criterionGAN(disc_outs[19], target_classes, True, is_gen=True)
        for disc_out in disc_outs:
            loss_G_GAN += self.criterionGAN(disc_out, target_classes, True, is_gen=True)
        print("D loss:", loss_G_GAN.mean())
        # new loss
        loss_G_latent = self.latent_reconst_criterion(l1, l2) * self.opt.lambda_latent
        loss_G_latent += self.latent_reconst_criterion(l3, l4) * self.opt.lambda_latent
        print("terminal latent code loss: ", loss_G_latent.mean(), self.opt.lambda_latent)
        # overall loss
        loss_G = (loss_G_GAN + loss_G_Rec + loss_G_Cycle +
                  loss_G_identity_reconst + loss_G_age_reconst + loss_G_latent + age_bias_loss).mean()
        loss_G.backward()
        self.optimizer_G.step()

        # update exponential moving average
        if self.use_moving_avg:
            self.accumulate(self.g_running, self.netG)

        # generate images for visdom
        if infer:
            if self.use_moving_avg:
                with torch.no_grad():
                    orig_id_features_out, _ = self.g_running.encode(self.reals)
                    # within domain decode
                    if self.opt.lambda_rec >= 0:
                        rec_images_out = self.g_running.decode(orig_id_features_out, self.orig_conditions)

                    # cross domain decode
                    gen_images_out = self.g_running.decode(orig_id_features_out, self.gen_conditions)
                    # encode generated
                    fake_id_features_out, _ = self.g_running.encode(gen_images[19])
                    # decode generated
                    if self.opt.lambda_cyc >= 0:
                        cyc_images_out = self.g_running.decode(fake_id_features_out, self.cyc_conditions)
            else:
                gen_images_out = gen_images[19]
                if self.opt.lambda_rec >= 0:
                    rec_images_out = rec_images[19]
                if self.opt.lambda_cyc >= 0:
                    cyc_images_out = cyc_images[19]

        loss_dict = {'loss_G_Adv': loss_G_GAN.mean(), 'loss_G_Cycle': loss_G_Cycle.mean(),
                     'loss_G_Rec': loss_G_Rec.mean(), 'loss_G_identity_reconst': loss_G_identity_reconst.mean(),
                     'loss_G_age_reconst': loss_G_age_reconst.mean(), "loss_G_latent": loss_G_latent.mean(),
                     'age_bias_loss': age_bias_loss.mean()}

        return [loss_dict,
                None if not infer else self.reals,
                None if not infer else gen_images_out,
                None if not infer else rec_images_out,
                None if not infer else cyc_images_out]

    def update_D(self):
        with torch.no_grad():
            # Discriminator optimization setp
            # set_trace()
            self.optimizer_D.zero_grad()
            self.get_conditions()

            ############### multi GPU ###############
            _, gen_images, _, _, _, _, _, _, _, _, _ = self.netG(self.reals, self.gen_conditions, None, None,
                                                                 disc_pass=True)

            # fake discriminator pass
            fake_disc_in = gen_images[19].detach()
            # with torch.no_grad():
            fake_disc_out = self.netD(fake_disc_in)

            # real discriminator pass
            real_disc_in = self.reals

            # necessary for R1 regularization
            real_disc_in.requires_grad_()
            # with torch.no_grad():
            real_disc_out = self.netD(real_disc_in)

            # Fake GAN loss
            fake_target_classes = torch.cat((self.class_B, self.class_A), 0)
            loss_D_fake = self.criterionGAN(fake_disc_out, fake_target_classes, False, is_gen=False)

            # Real GAN loss
            real_target_classes = torch.cat((self.class_A, self.class_B), 0)
            loss_D_real = self.criterionGAN(real_disc_out, real_target_classes, True, is_gen=False)
            loss_D = (loss_D_fake + loss_D_real).mean()
            # loss_D.backward()
        return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean()}
        # R1 regularization
        # loss_D_reg = self.R1_reg(real_disc_out, real_disc_in)
        #
        # loss_D = (loss_D_fake + loss_D_real + loss_D_reg).mean()
        # loss_D.backward()
        # # self.optimizer_D.step()
        # 6
        # return {'loss_D_real': loss_D_real.mean(), 'loss_D_fake': loss_D_fake.mean(), 'loss_D_reg': loss_D_reg.mean()}

    def inference(self, data):
        self.set_inputs(data, mode='test')
        if self.isEmpty:
            return

        self.numValid = self.valid.sum().item()
        sz = self.reals.size()
        self.fake_B = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])
        self.cyc_A = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])

        with torch.no_grad():
            if self.traverse or self.deploy:
                if self.traverse and self.compare_to_trained_outputs:
                    start = self.compare_to_trained_class - self.trained_class_jump
                    end = start + (
                            self.trained_class_jump * 2) * 2  # arange is between [start, end), end is always omitted
                    self.class_B = torch.arange(start, end, step=self.trained_class_jump * 2, dtype=self.class_A.dtype)
                else:
                    self.class_B = torch.arange(self.numClasses, dtype=self.class_A.dtype)

                self.get_conditions(mode='test')

                self.fake_B = self.netG.infer(self.reals, self.gen_conditions, traverse=self.traverse,
                                              deploy=self.deploy, interp_step=self.opt.interp_step)
            else:
                for i in range(self.numClasses):
                    self.class_B = self.Tensor(self.numValid).long().fill_(i)
                    self.get_conditions(mode='test')

                    if self.isTrain:
                        self.fake_B[i, :, :, :, :] = self.g_running.infer(self.reals, self.gen_conditions)
                    else:
                        # st()
                        # print(type(self.netG.infer(self.reals, self.gen_conditions)))
                        self.fake_B[i, :, :, :, :] = self.netG.infer(self.reals, self.gen_conditions)[0][5]

                    cyc_input = self.fake_B[i, :, :, :, :]

                    if self.isTrain:
                        self.cyc_A[i, :, :, :, :] = self.g_running.infer(cyc_input, self.cyc_conditions)
                    else:
                        self.cyc_A[i, :, :, :, :] = self.netG.infer(cyc_input, self.cyc_conditions)[0][5]

            visuals = self.get_visuals()

        return visuals

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.use_moving_avg:
            self.save_network(self.g_running, 'g_running', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lr = self.old_lr * self.opt.decay_gamma
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            mult = param_group.get('mult', 1.0)
            param_group['lr'] = lr * mult
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_visuals(self):
        return_dicts = [OrderedDict() for i in range(self.numValid)]

        real_A = util.tensor2im(self.reals.data)
        fake_B_tex = util.tensor2im(self.fake_B.data)

        if self.debug_mode:
            rec_A_tex = util.tensor2im(self.cyc_A.data[:, :, :, :, :])

        if self.numValid == 1:
            real_A = np.expand_dims(real_A, axis=0)

        for i in range(self.numValid):
            # get the original image and the results for the current samples
            curr_real_A = real_A[i, :, :, :]
            real_A_img = curr_real_A[:, :, :3]

            # start with age progression/regression images
            if self.traverse or self.deploy:
                curr_fake_B_tex = fake_B_tex
                orig_dict = OrderedDict([('orig_img', real_A_img)])
            else:
                curr_fake_B_tex = fake_B_tex[:, i, :, :, :]
                orig_dict = OrderedDict([('orig_img_cls_' + str(self.class_A[i].item()), real_A_img)])

            return_dicts[i].update(orig_dict)

            # set output classes numebr
            if self.traverse:
                out_classes = curr_fake_B_tex.shape[0]
            else:
                out_classes = self.numClasses

            for j in range(out_classes):
                fake_res_tex = curr_fake_B_tex[j, :, :, :3]
                fake_dict_tex = OrderedDict([('tex_trans_to_class_' + str(j), fake_res_tex)])
                return_dicts[i].update(fake_dict_tex)

            if not (self.traverse or self.deploy):
                if self.debug_mode:
                    # continue with tex reconstructions
                    curr_rec_A_tex = rec_A_tex[:, i, :, :, :]
                    orig_dict = OrderedDict([('orig_img2', real_A_img)])
                    return_dicts[i].update(orig_dict)
                    for j in range(self.numClasses):
                        rec_res_tex = curr_rec_A_tex[j, :, :, :3]
                        rec_dict_tex = OrderedDict([('tex_rec_from_class_' + str(j), rec_res_tex)])
                        return_dicts[i].update(rec_dict_tex)

        return return_dicts


class InferenceModel(LATS):
    def forward(self, data):
        return self.inference(data)
