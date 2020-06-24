import matplotlib; matplotlib.use('Agg')
import argparse
import copy
import glob
import importlib.machinery
import importlib.util
import multiprocessing as mp
import os
import random
import re
from multiprocessing import Pool, Manager

import numpy as np
from blox import AttrDict
from gcp.infra.sim.benchmarks import run_trajectories


def bench_worker(conf, iex=-1, ngpu=1):
    print('started process with PID:', os.getpid())
    random.seed(None)
    np.random.seed(None)
    print('start ind', conf['start_index'])
    print('end ind', conf['end_index'])
    run_trajectories(conf, iex, gpu_id=conf['gpu_id'], ngpu=ngpu)


def check_and_pop(dict, key):
    if dict.pop(key, None) is not None:
        print('popping key: {}'.format(key))


def postprocess_hyperparams(hyperparams, args):
    if args.data_save_postfix:
        hyperparams['data_save_dir'] = os.path.join(hyperparams['data_save_dir'], args.data_save_postfix)
    return hyperparams


class ControlManager:
    def __init__(self, args_in=None, hyperparams=None):
        parser = argparse.ArgumentParser(description='run parallel data collection')
        parser.add_argument('experiment', type=str, help='experiment name')
        parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
        parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
        parser.add_argument('--ngpu', type=int, help='the number of gpus to use', default=1)
        parser.add_argument('--gpu', type=int, help='the gpu to use', default=-1)
        parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
        parser.add_argument('--isplit', type=int, help='split id', default=-1)
        parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)
        parser.add_argument('--data_save_postfix', type=str, help='appends to the data_save_dir path', default='')
        parser.add_argument('--nstart_goal_pairs', type=int, help='max number of start goal pairs', default=None)
        parser.add_argument('--resume_from', type=int, help='from which traj idx to continue generating', default=None)

        args = parser.parse_args(args_in)

        print("Resume from")
        print(args.resume_from)

        if args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        if hyperparams is None:
            hyperparams_file = args.experiment
            loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            hyperparams = AttrDict(mod.config)

        self.args = args
        self.hyperparams = postprocess_hyperparams(hyperparams, args)
        
    def run(self, logging_conf=None):
        args = self.args
        hyperparams = self.hyperparams
    
        gpu_id = args.gpu_id
    
        n_worker = args.nworkers
        if args.nworkers == 1:
            parallel = False
        else:
            parallel = True
        print('parallel ', bool(parallel))

        if args.nsplit != -1:
            assert args.isplit >= 0 and args.isplit < args.nsplit, "isplit should be in [0, nsplit-1]"
           
            n_persplit = max((hyperparams['end_index'] + 1 - hyperparams['start_index']) / args.nsplit, 1)
            hyperparams['end_index'] = int((args.isplit + 1) * n_persplit + hyperparams['start_index'] - 1)
            hyperparams['start_index'] = int(args.isplit * n_persplit + hyperparams['start_index'])

        n_traj = hyperparams['end_index'] - hyperparams['start_index'] + 1
        traj_per_worker = int(n_traj // np.float32(n_worker))
        offset = int(args.resume_from // np.float32(n_worker)) if args.resume_from is not None else 0
        start_idx = [hyperparams['start_index'] + offset + traj_per_worker * i for i in range(n_worker)]
        end_idx = [hyperparams['start_index'] + traj_per_worker * (i+1)-1 for i in range(n_worker)]

        if 'gen_xml' in hyperparams['agent']:
            try:
                os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
            except: pass

        self.set_paths(hyperparams, args)
        record_queue, record_saver_proc, counter = None, None, None

        if args.iex != -1:
            hyperparams['agent']['iex'] = args.iex
    
        conflist = []
        for i in range(n_worker):
            modconf = copy.deepcopy(hyperparams)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]
            modconf['ntraj'] = n_traj
            modconf['gpu_id'] = i + gpu_id
            if logging_conf is not None:
                modconf['logging_conf'] = logging_conf
            conflist.append(modconf)
        if parallel:
            self.start_parallel(conflist, n_worker)
        else:
            bench_worker(conflist[0], args.iex, args.ngpu)
    
        if args.save_thread:
            record_queue.put(None)           # send flag to background thread that it can end saving after it's done
            record_saver_proc.join()         # joins thread and continues execution

    def set_paths(self, hyperparams, args):
        subpath = str.partition(hyperparams['current_dir'], 'experiments')[-1]

        if 'data_save_dir' not in hyperparams:
            data_save_dir = os.environ['GCP_DATA_DIR'] + '/' + subpath
            hyperparams['data_save_dir'] = data_save_dir
        print('setting data_save_dir to', hyperparams['data_save_dir'])
        if 'log_dir' not in hyperparams:
            log_dir = os.environ['GCP_EXP_DIR'] + '/' + subpath
            if args.data_save_postfix:
                log_dir = os.path.join(log_dir, args.data_save_postfix)
            hyperparams['log_dir'] = log_dir
        print('setting log_dir to', hyperparams['log_dir'])
        result_dir = hyperparams['data_save_dir'] + '/log'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for file in glob.glob(result_dir + '/*.pkl'):
            os.remove(file)
        hyperparams['result_dir'] = result_dir

    def start_parallel(self, conflist, n_worker):
        # mp.set_start_method('spawn')  # this is important for parallelism with xvfb
        p = Pool(n_worker)
        p.map(bench_worker, conflist)


if __name__ == '__main__':
    ControlManager().run()
