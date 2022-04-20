### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
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
#from  ipdb import set_trace
import glob
def train(opt):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    if opt.continue_train:
        if opt.which_epoch == 'latest':
            try:
                start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
            except:
                start_epoch, epoch_iter = 1, 0
        else:
            start_epoch, epoch_iter = int(opt.which_epoch), 0

        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
        for update_point in opt.decay_epochs:
            if start_epoch < update_point:
                break

            opt.lr *= opt.decay_gamma
    else:
        start_epoch, epoch_iter = 0, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    for param in model.parameters():
        param.requires_grad = False
    # 解冻部分   # netG
    for param in model.g_running.decoder.mlp1.parameters():
        param.requires_grad = True
    for param in model.g_running.decoder.mlp2.parameters():
        param.requires_grad = True
    for param in model.g_running.decoder.func.parameters():
        param.requires_grad = True
    for param in model.netG.decoder.mlp1.parameters():
        param.requires_grad = True
    for param in model.netG.decoder.mlp2.parameters():
        param.requires_grad = True
    for param in model.netG.decoder.func.parameters():
        param.requires_grad = True
    model_dict = model.state_dict()

    # set_trace()
    # def load_total_work(model,pretrained_path):
    #     model_dict = model.state_dict()
    #     items=glob.glob(os.path.join(pretrained_path ,"latest*.pth"))
    #     for pretrained_dict_path in items:
    #         pretrained_dict = torch.load(pretrained_dict_path)
    #         for k, v in pretrained_dict.items():
    #             # set_trace()
    #             print("pretrain",k)
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    #         print('can be loaded',pretrained_dict)
    #         model.load_state_dict(pretrained_dict, strict=False)
    #     # from sets import Set
    #     not_initialized = set()
    #     # for k, v in model.state_dict().items():
    #     #     if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
    #     #         not_initialized.add(k.split('.')[0])
    #     print(sorted(not_initialized))
    # load_total_work(model,pretrained_path)
    # items=glob.glob(os.path.join(pretrained_path ,"*.pth"))
    # print(items)
    # for pretrained_dict_path in items:
    #     pretrained_dict = torch.load(pretrained_dict_path)
    #     for k, v in pretrained_dict.items():
    #         print("pretrain",k)
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)  # 用预训练模型参数更新new_model中的部分参数
    #     model.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中
    # st()
    visualizer = Visualizer(opt)

    total_steps = (start_epoch) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    bSize = opt.batchSize

    #in case there's no display sample one image from each class to test after every epoch
    if opt.display_id == 0:
        dataset.dataset.set_sample_mode(True)
        dataset.num_workers = 1
        for i, data in enumerate(dataset):
            if i*opt.batchSize >= opt.numClasses:
                break
            if i == 0:
                sample_data = data
            else:
                for key, value in data.items():
                    if torch.is_tensor(data[key]):
                        sample_data[key] = torch.cat((sample_data[key], data[key]), 0)
                    else:
                        sample_data[key] = sample_data[key] + data[key]
        dataset.num_workers = opt.nThreads
        dataset.dataset.set_sample_mode(False)

    for epoch in range(start_epoch, opt.epochs):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = 0
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = (total_steps % opt.display_freq == display_delta) and (opt.display_id > 0)

            ############## Network Pass ####FREC####################
            # set_trace()
            model.set_inputs(data)
            disc_losses = model.update_D()
            # st()
            gen_losses, gen_in, gen_out, rec_out, cyc_out = model.update_G(infer=save_fake)
            print("yzb",i)
            loss_dict = dict(gen_losses, **disc_losses)
            ##################################################
            print('wxq', gen_losses['age_bias_loss'])
            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:

                errors = {k: v.item() if not (isinstance(v, float) or isinstance(v, int)) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch+1, epoch_iter, errors, t)

                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            ### display output images
            # if save_fake and opt.display_id > 0:
            #     class_a_suffix = ' class {}'.format(data['A_class'][0])
            #     class_b_suffix = ' class {}'.format(data['B_class'][0])
            #     classes = None
            #
            #     visuals = OrderedDict()
            #     visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[0]))])
            #     visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[bSize]))])
            #
            #     A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, util.tensor2im(gen_out.data[0]))])
            #     B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, util.tensor2im(gen_out.data[bSize]))])
            #     if opt.lambda_rec > 0:
            #         A_out_vis.update([('reconstructed image' + class_a_suffix, util.tensor2im(rec_out.data[0]))])
            #         B_out_vis.update([('reconstructed image' + class_b_suffix, util.tensor2im(rec_out.data[bSize]))])
            #     if opt.lambda_cyc > 0:
            #         A_out_vis.update([('cycled image' + class_a_suffix, util.tensor2im(cyc_out.data[0]))])
            #         B_out_vis.update([('cycled image' + class_b_suffix, util.tensor2im(cyc_out.data[bSize]))])
            #
            #     visuals_A.update(A_out_vis)
            #     visuals_B.update(B_out_vis)
            #     visuals.update(visuals_A)
            #     visuals.update(visuals_B)
            #
            #     ncols = len(visuals_A)
            #     visualizer.display_current_results(visuals, epoch, classes, ncols)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch+1, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
                if opt.display_id == 0:
                    model.eval()
                    visuals = model.inference(sample_data)
                    visualizer.save_matrix_image(visuals, 'latest')
                    model.train()

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.epochs, time.time() - epoch_start_time))

        ### save model for this epoch
        if (epoch+1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch+1, total_steps))
            model.save('latest')
            model.save(epoch+1)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            if opt.display_id == 0:
                model.eval()
                visuals = model.inference(sample_data)
                visualizer.save_matrix_image(visuals, epoch+1)
                model.train()

        ### multiply learning rate by opt.decay_gamma after certain iterations
        if (epoch+1) in opt.decay_epochs:
            model.update_learning_rate()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
