from __future__ import absolute_import
from __future__ import print_function
import code
import os
import sys
import numpy as np

sys.path.append('/media/ben/datadrive/Software/VGG-Speaker-Recognition/src')
sys.path.append('/media/ben/datadrive/Software/VGG-Speaker-Recognition/tool')
import toolkits
import utils as ut

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'],
                    type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
#  /media/ben/datadrive/Software/VGG-Speaker-Recognition/model/gvlad_softmax
#  /resnet34_vlad8_ghost2_bdim512_deploy/weights.h5, EER: 0.483208955223

#  EER: 0.0308370044053
# 2020-11-15_resnet34s_bs16_adam_lr0.001_vlad8_ghost2_bdim512_ohemlevel0/weights-42-0.931.h5

arguments = "--net resnet34s --gpu 0 --ghost_cluster 2 --vlad_cluster 8 --loss softmax --resume " \
            "/media/ben/datadrive/Software/VGG-Speaker-Recognition/model/gvlad_softmax/2020-11-15_resnet34s_bs16_adam_lr0.001_vlad8_ghost2_bdim512_ohemlevel0/weights-42-0.931.h5 --data_path " \
            "/media/ben/datadrive/Zalo/voice-verification/Train-Test-Data/dataset/".split()

ZALO_TEST = "/media/ben/datadrive/Zalo/voice-verification/vgg_db_files/val_trials.txt"

args = parser.parse_args(arguments)


def main():
    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    if args.test_type == 'normal':
        verify_list = np.loadtxt(ZALO_TEST, str)
    elif args.test_type == 'hard':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_hard.txt', str)
    elif args.test_type == 'extend':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            result_path = set_result_path(args)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    code.interact(local=locals())

    print("gathering features")
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(unique_list):
        if c % 10 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'], hop_length=params['hop_length'], n_fft=params['nfft'], spec_len=params['spec_len'], mode='eval')
        # specs.shape == (257, 1117)
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        # specs.shape ==  (1, 257, 1117, 1)
        v = network_eval.predict(specs)
        # v.shape == (1, 512)
        feats += [v]

    feats = np.array(feats)

    print("Computing similarities")
    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        # scores : 0.480307102203, gt : 0
        # scores : 0.613937497139, gt : 0
        # scores : 0.456035703421, gt : 0
        # scores : 0.537688136101, gt : 0
        # scores : 0.432827711105, gt : 0
        # scores : 0.368318378925, gt : 0
        # scores : 0.465793728828, gt : 0
        # scores : 0.625230908394, gt : 0
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]
        # from the hash table of unique speaker embeddings, extract relevant items
        v1 = feats[ind1, 0] # shape.(512,)
        v2 = feats[ind2, 0]

        scores += [np.sum(v1 * v2)] # v1 * v2 0.4803071

        labels += [verify_lb[c]]
        print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    np.save(os.path.join(result_path, 'prediction_scores.npy'), scores)
    np.save(os.path.join(result_path, 'groundtruth_labels.npy'), labels)

    eer, thresh = toolkits.calculate_eer(labels, scores)
    print('==> model : {}, EER: {}'.format(args.resume, eer))


def set_result_path(args):
    model_path = args.resume
    exp_path = model_path.split(os.sep)
    result_path = os.path.join('../result', exp_path[2], exp_path[3])
    if not os.path.exists(result_path): os.makedirs(result_path)
    return result_path


if __name__ == "__main__":
    main()
