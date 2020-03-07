# coding=utf-8
# Copyright 2020 Heewon Jeon. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from utils.embedding_maker import create_embeddings


parser = argparse.ArgumentParser(description='Korean Autospacing Embedding Maker')

parser.add_argument('--num-iters', type=int, default=5,
                    help='number of iterations to train (default: 5)')

parser.add_argument('--min-count', type=int, default=100,
                    help='mininum word counts to filter (default: 100)')

parser.add_argument('--embedding-size', type=int, default=100,
                    help='embedding dimention size (default: 100)')

parser.add_argument('--num-worker', type=int, default=16,
                    help='number of thread (default: 16)')

parser.add_argument('--window-size', type=int, default=8,
                    help='skip-gram window size (default: 8)')

parser.add_argument('--corpus_dir', type=str, default='data',
                    help='training resource dir')

parser.add_argument('--train', action='store_true', default=True,
                    help='do embedding trainig (default: True)')

parser.add_argument('--model-file', type=str, default='kospacing_wv.mdl',
                    help='output object from Word2Vec() (default: kospacing_wv.mdl)')

parser.add_argument('--numpy-wv', type=str, default='kospacing_wv.np',
                    help='numpy object file path from Word2Vec() (default: kospacing_wv.np)')

parser.add_argument('--w2idx', type=str, default='w2idx.dic',
                    help='item to index json dictionary (default: w2idx.dic)')

parser.add_argument('--model-dir', type=str, default='model',
                    help='dir to save models (default: model)')

opt = parser.parse_args()

if opt.train:
    create_embeddings(opt.corpus_dir, opt.model_dir + '/' +
                      opt.model_file, opt.model_dir + '/' + opt.numpy_wv,
                      opt.model_dir + '/' + opt.w2idx, min_count=opt.min_count,
                      iter=opt.num_iters,
                      size=opt.embedding_size, workers=opt.num_worker, window=opt.window_size)
