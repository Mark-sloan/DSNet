import sys
sys.path.append('../..')
import os
import tensorflow as tf
import argparse
from tensorpack import *
from models.shuffle_tensorpack import ShuffleTensorpack
import datetime
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils import argscope, get_model_loader, model_utils, get_global_step_value
from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.models import regularize_cost
from tensorpack.utils import logger
import multiprocessing
import numpy as np
from cityscapes import CITYSCAPES



def get_city_dataflow(name, batch_size, sub_rate, is_train, random_crop, parallel=None):
    '''
    Get cityscapes dataflow.
    '''
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)
    
    ds = CITYSCAPES(name, sub_rate, shuffle=is_train, random_crop=random_crop)    
    ds = PrefetchDataZMQ(ds, parallel)
    ds = BatchData(ds, batch_size, remainder=False)
    
    return ds
        
        
def get_config(nr_tower, args):
    
    TOTAL_BATCH_SIZE = args.batch_size
    batchsize = TOTAL_BATCH_SIZE // nr_tower
    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batchsize))
    
    max_epoch = args.num_epochs
    lr = args.initial_learning_rate
    num_epochs_before_decay = args.num_epochs_before_decay
    decay_factor = args.decay_factor
    num_decay = int(max_epoch/num_epochs_before_decay)
    
    if args.dataset_mode == 'train_fine':
        dataset_size = 2975
    elif args.dataset_mode == 'validation_fine':
        dataset_size = 500
    elif args.dataset_mode == 'train_patches':
        dataset_size = 2975*8 #23800
    elif args.dataset_mode == 'validation_patches':
        dataset_size = 500*8
    elif args.dataset_mode == 'train_coarse':
        dataset_size = 14440
    elif args.dataset_mode == 'combine_patches':
        dataset_size = 23800 + 3000
    elif args.dataset_mode == 'combine_val_patches':
        dataset_size = 1000
        
    steps_per_epoch = int(dataset_size / TOTAL_BATCH_SIZE)
    max_iter = max_epoch * steps_per_epoch
    
    schedule=[]
    if args.lr_type == 'poly':
        end_lr = 2e-5
        for i in range(max_epoch):
            ep = i 
            val = (lr - end_lr) * np.power((1 - 1.*i / num_epochs_before_decay), 0.9) + end_lr
            schedule.append((ep, val))
    if args.lr_type == 'exponential_decay':
        for i in range(num_decay):
            ep = i * num_epochs_before_decay
            val = lr * np.power(decay_factor, i)
            schedule.append((ep, val))
    
    model = ShuffleTensorpack(args, sub_rate=args.sub_rate, batchsize=batchsize)
    
    dataset_train = get_city_dataflow(args.dataset_mode, batchsize, args.sub_rate, 
                                      is_train=True, random_crop=args.random_crop)
      
    logger.set_logger_dir(os.path.join('log', args.exp_name+'_'+str(datetime.date.today())))
    checkpoint_dir = os.path.join('log', args.exp_name+'_'+str(datetime.date.today()),'save')
    infs = [ScalarStats(names='mean_iou', prefix='val')] # val_mean_IoU
    callbacks = [
        PeriodicTrigger(ModelSaver(max_to_keep=5, checkpoint_dir=checkpoint_dir),every_k_steps=250),       
        ScheduledHyperParamSetter('learning_rate', schedule=schedule),
        EstimatedTimeLeft(),
        MergeAllSummaries(period=250),
        ]
    
    if args.save_val_max is True:
        dataset_val = get_city_dataflow(args.dataset_val_mode, TOTAL_BATCH_SIZE, args.sub_rate, 
                                        is_train=False, random_crop=args.random_crop)     
        callbacks.extend([PeriodicTrigger(DataParallelInferenceRunner(dataset_val, infs, [0,1,2,3]),every_k_steps=250),
                         PeriodicTrigger(MaxSaver(monitor_stat='val_mean_iou', checkpoint_dir=checkpoint_dir),every_k_steps=250)])
    
    return AutoResumeTrainConfig(model=model,
                                dataflow=dataset_train,
                                callbacks=callbacks,
                                steps_per_epoch=steps_per_epoch,
                                max_epoch=max_epoch,)
    
#    return TrainConfig(
#        model=model,
#        dataflow=dataset_train,
#        callbacks=callbacks,
#        steps_per_epoch=steps_per_epoch,
#        max_epoch=max_epoch,
#    )
    
    

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_name',
                        help='experiment name, should reflect main feature and date')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Train batch size')
    parser.add_argument('--batch_size_val', type=int, default=4,
                        help='Validation batch, can be larger than train batch size')
    parser.add_argument('--dataset_mode', type=str, default='train_fine',
                        help='train_fine, combine_fine, test_fine, train_coarse, patches')
    parser.add_argument('--dataset_val_mode', type=str, default='validation_fine',
                        help='validation_fine, patches_val')
    parser.add_argument('--initial_learning_rate', type=float, default=5e-4,
                        help='initial_learning_rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='l2 weight_decay')    
    parser.add_argument('--num_groups', type=int, default=4,
                        help='number of groups for channel shuffle')
    parser.add_argument('--num_groups_norm', type=int, default=64, 
                        help='num of groups for group norm')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr_type', type=str, default='exponential_decay',
                        help='learning rate type')
    parser.add_argument('--num_epochs_before_decay', type=int, default=100,
                        help='number of epochs before decay leraning rate')
    parser.add_argument('--decay_factor', type=float, default=0.2,
                        help='decay factor when learning rate decay happens')
    parser.add_argument('--loss_mode', type=str, default='',
                        help='normal loss, focal loss')
    parser.add_argument('--loss_type', type=str, default='small2big',
                        help='three types of loss: small2big, small2small, big2big')
    parser.add_argument('--sub_rate', type=int, default=1,
                        help='model input subsample rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='which type optimizer')    
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain file addr')
    parser.add_argument('--crop', type=bool, default=False,
                        help='crop patches')
    parser.add_argument('--add_se_loss', type=bool, default=False,
                        help='use semantic encoding loss')
    parser.add_argument('--save_val_max', type=bool, default=False,
                        help='whether inference during training and save max performance')
    parser.add_argument('--gpu', type=str, default='0',
                        help='which gpu to use')
    parser.add_argument('--random_crop', type=bool, default=False,
                        help='use semantic encoding loss')
    args = parser.parse_args()
    print('random crop:', args.random_crop)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    
    nr_tower = max(get_num_gpu(), 1)
    config = get_config(nr_tower, args)
    
    if args.pretrain:
        pretrain_ckpt = tf.train.latest_checkpoint(args.pretrain)
        #pretrain_ckpt = args.pretrain+'/model-593500'
        #pretrain_ckpt = args.pretrain+'/max-val_mean_iou'
        config.session_init = SaverRestore(model_path=pretrain_ckpt, ignore=['Stage3encoding', 'encoding/fc3', 'labels_vector'])
        #config.session_init = get_model_loader(pretrain_ckpt)
        #loader = tf.train.Saver(var_list = [v for v in tf.global_variables() if 'encoding' not in v.name])

    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower)) 
    # This trainer is not working with batchnorm sync
    # ref: https://github.com/tensorpack/tensorpack/issues/912#issuecomment-425156860
    
    #launch_train_with_config(config, SyncMultiGPUTrainerReplicated([0,1,2,3], mode='nccl'))
    
    
    
    
    
    
        

        