import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import math
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import UnlearningTrainer, RelearningTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

class ULRL(defense):

    def __init__(self):
        super(ULRL).__init__()
        pass

    def set_args(self, parser):
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument('--frequency_save', type=int, help='0 is never')
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("--dataset_path", type=str)

        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--lr_scheduler', type=str)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=int, nargs='+')
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)

        # Unlearning
        parser.add_argument('--unlearning_epochs', type=int)
        parser.add_argument('--unlearning_lr', type=float)
        parser.add_argument('--clean_threshold', type=float)

        # Relearning
        parser.add_argument('--init', type=bool)
        parser.add_argument('--relearn_epochs', type=int)
        parser.add_argument('--relearn_lr', type=float)
        parser.add_argument('--relearn_alpha', type=float)
        parser.add_argument('--linear_name', type=str)

        # Backdoor Attacks
        parser.add_argument('--attack', type=str)
        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)
        parser.add_argument('--trigger_type', type=str,
                            help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ulrl/config.yaml", help='the path of yaml')
        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        defense_save_path = "record" + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "ulrl"
        os.makedirs(defense_save_path, exist_ok = True)

        args.defense_save_path = defense_save_path
        return args

    def prepare(self, args):

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            args.defense_save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log.
        logger.setLevel(0)
        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        fix_random(args.random_seed)
        self.args = args

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
        '''

        self.attack_result = load_attack_result("record" + os.path.sep + self.args.result_file + os.path.sep +'attack_result.pt')
        netC = generate_cls_model(args.model, args.num_classes)
        netC.load_state_dict(self.attack_result['model'])
        netC.to(args.device)
        self.netC = netC                  

    def defense(self):
        
        netC = self.netC
        args = self.args
        attack_result = self.attack_result

        # clean_train with subset
        clean_train_dataset_with_transform = attack_result['clean_train']
        clean_train_dataset_without_transform = clean_train_dataset_with_transform.wrapped_dataset
        clean_train_dataset_without_transform = prepro_cls_DatasetBD_v2(
            clean_train_dataset_without_transform
        )
        ran_idx = choose_index(args, len(clean_train_dataset_without_transform))
        clean_train_dataset_without_transform.subset(ran_idx)
        clean_train_dataset_with_transform.wrapped_dataset = clean_train_dataset_without_transform
        log_index = args.defense_save_path + os.path.sep + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        trainloader = torch.utils.data.DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=True)

        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform
        clean_test_dataloader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                        pin_memory=args.pin_memory)

        bd_test_dataset_with_transform = attack_result['bd_test']
        data_bd_testset = bd_test_dataset_with_transform
        bd_test_dataloader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                     pin_memory=args.pin_memory)

        #################################################################################
        ##################                 UNLEARNING                 ###################
        #################################################################################

        unlearning_start = time.time()
        logging.info('----------- UNLEARNING --------------')
        netC.train()
        netC.requires_grad_()
        args.lr = args.unlearning_lr
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = argparser_opt_scheduler(netC, self.args)
        
        pre_nu_weights = getattr(netC, args.linear_name).weight.data.clone().detach().to('cpu')
        unlearning_trainer = UnlearningTrainer(netC, args.clean_threshold)
        unlearning_trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            clean_test_dataloader,
            bd_test_dataloader,
            args.unlearning_epochs,
            criterion,
            optimizer,
            scheduler,
            args.amp,
            torch.device(args.device),
            args.frequency_save,
            self.args.defense_save_path,
            "unlearn",
            prefetch=False,
            prefetch_transform_attr_name="transform",
            non_blocking=args.non_blocking,
        )

        logging.info('----------- Filter Backdoor Neuron with MAD --------------')
        # Absolute changes for each weight and aggregate  changes for each neuron
        post_nu_weights = getattr(netC, args.linear_name).weight.data.clone().detach().to('cpu') 
        absolute_weight_changes = (post_nu_weights - pre_nu_weights).abs()
        aggregate_changes_per_neuron = absolute_weight_changes.sum(dim=1)

        changes_str = ", ".join([f"{i}: {change.item():.2f}" for i, change in enumerate(aggregate_changes_per_neuron)])
        logging.info(f"Neuron Weight changes: [{changes_str}]")

        # Calculate MAD (Median Absolute Deviation)
        median = torch.median(aggregate_changes_per_neuron)
        mad = torch.median(torch.abs(aggregate_changes_per_neuron - median))
        mad_z_scores = torch.abs((aggregate_changes_per_neuron - median) / (1.4826 * mad))
        neuron_indices = torch.where((mad_z_scores > 3.5) & (aggregate_changes_per_neuron > median))[0].tolist()
        
        # Candidate Neuron Length Threshold 
        candidate_neuron_length = 2 
        if len(neuron_indices) > candidate_neuron_length:
            neuron_indices = neuron_indices[:candidate_neuron_length]

        logging.info(f"Indices of neurons with maximal changes based on MAD: {neuron_indices}")
        unlearning_end = time.time()
        logging.info(f"Unlearning Time Cost: {unlearning_end-unlearning_start:.2f}s")

        #################################################################################
        ##################                 RELEARNING                 ###################
        #################################################################################
        logging.info('----------- RELEARNING --------------')
        relearning_start = time.time()
        netC.load_state_dict(self.attack_result['model']) 

        weight_mat_ori = getattr(netC, args.linear_name).weight.data.clone().detach()[neuron_indices, :] 

        param_list = []
        for name, param in netC.named_parameters():
            if args.linear_name in name:
                if args.init:       # He initialization
                    std = math.sqrt(2. / param.size(-1)) 
                    if 'weight' in name:
                        logging.info(f'Initialize linear classifier {name}.')
                        for idx in neuron_indices:
                            param.data[idx].uniform_(-std, std)

                    elif 'bias' in name:
                        logging.info(f'Initialize linear classifier {name}.')
                        for idx in neuron_indices:
                            param.data[idx].fill_(0)
            
            param.requires_grad = True
            param_list.append(param)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(param_list, lr=args.relearn_lr, momentum = 0.9) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.relearn_epochs)
    
        relearning_trainer = RelearningTrainer(netC, args.linear_name, neuron_indices, 
                                                 weight_mat_ori, args.relearn_alpha)
        
        relearning_trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            clean_test_dataloader,
            bd_test_dataloader,
            args.relearn_epochs,
            criterion,
            optimizer,
            scheduler,
            args.amp,
            torch.device(args.device),
            args.frequency_save,
            self.args.defense_save_path,
            "relearn",
            prefetch=False,
            prefetch_transform_attr_name="transform",
            non_blocking=args.non_blocking,
        )        
        relearning_trainer.test_on_mix()

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = netC.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )
        relearning_end = time.time()
        logging.info(f"Relearning Time Cost: {relearning_end-relearning_start:.2f}s") 

if __name__ == '__main__':
    ulrl = ULRL()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = ulrl.set_args(parser)
    args = parser.parse_args()
    ulrl.add_yaml_to_args(args)
    args = ulrl.process_args(args)
    ulrl.prepare(args)
    start = time.time()
    ulrl.defense()
    end = time.time()
    logging.info(f"Time cost: {end-start:.2f}s")