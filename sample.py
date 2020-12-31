#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle


from six import text_type
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from model import Model

parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store checkpointed models')
parser.add_argument('--save_path', type=str, default='data',
                    help='output results directory')
parser.add_argument('--dataset', type=str, default='data',
                    help='dataset')
parser.add_argument('-n', type=int, default=500,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u'',
                    help='prime text')
parser.add_argument('--sample', type=int, default=1,
                    help='0 to use max at each timestep, 1 to sample at '
                         'each timestep, 2 to sample on spaces')
parser.add_argument('--threshold', type=float, default=0,
                    help='set up threshold of probability')
parser.add_argument('--task', type=str, default=0,
                    help='sample, evaluate, evaluate2')
parser.add_argument('--seq_length', type=int, default=0,
                    help='seq_length')
args = parser.parse_args()


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
   
    model = Model(saved_args, 'sample')
    
    if args.task == 'prime':
        chars, ret = model.sample2(vocab, args.prime, args.save_path, args.save_dir)
        data = np.array([chars, ret]).T
        print(data)
    if args.task == 'corpus_all':
        with open('./data/'+args.dataset+'/test.txt') as f:
            lines = f. readlines()[0]
            
        chars, ret, p, c, h = model.evaluate(vocab, lines, args.save_path, args.save_dir)
        df = {'char':chars,'nextP':ret, 'probs':p, 'cells':c, 'hidden': h}
        df = pd.DataFrame(df)
        df = df[(df['char']!='^') & (df['char']!='$')]
        pickle.dump(df, open('./data/'+args.dataset+'/test/'+args.save_path+'.p', "wb" ))


if __name__ == '__main__':
    sample(args)
