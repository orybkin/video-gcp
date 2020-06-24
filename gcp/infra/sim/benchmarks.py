from .simulator import Sim
import os
import numpy as np
import pickle
from collections import OrderedDict
from .util.combine_score import write_scores
from gcp.infra.agent.benchmarking_agent import BenchmarkAgent
import pdb


def run_trajectories(conf=None, iex=-1, gpu_id=None, ngpu=1):
    """
    :param conf:
    :param iex:  if not -1 use only rollout this example
    :param gpu_id:
    :return:
    """
    log_dir= conf['log_dir']

    if conf != None:
        benchmark_name = 'parallel'

    print('-------------------------------------------------------------------')
    print('name of algorithm setting: ' + benchmark_name)
    print('agent settings')
    for key in list(conf['agent'].keys()):
        print(key, ': ', conf['agent'][key])
    print('------------------------')
    print('------------------------')
    print('policy settings')
    for key in list(conf['policy'].keys()):
        print(key, ': ', conf['policy'][key])
    print('-------------------------------------------------------------------')

    # sample intial conditions and goalpoints

    sim = Sim(conf, gpu_id=gpu_id, ngpu=ngpu)

    if iex == -1:
        i_traj = conf['start_index']
        nruns = conf['end_index']
        print('started worker going from ind {} to in {}'.format(conf['start_index'], conf['end_index']))
    else:
        i_traj = iex
        nruns = iex

    stats_lists = OrderedDict()

    if 'sourcetags' in conf:  # load data per trajectory
        if 'VMPC_DATA_DIR' in os.environ:
            datapath = conf['source_basedirs'][0].partition('pushing_data')[2]
            conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] + datapath]

    result_file = log_dir + '/results_{}to{}.txt'.format(conf['start_index'], conf['end_index'])
    final_dist_pkl_file = log_dir + '/scores_{}to{}.pkl'.format(conf['start_index'], conf['end_index'])
    if os.path.isfile(log_dir + '/result_file'):
        raise ValueError("the file {} already exists!!".format(result_file))

    while i_traj <= nruns:

        print('run number ', i_traj)
        print('loading done')

        print('-------------------------------------------------------------------')
        print('run number ', i_traj)
        print('-------------------------------------------------------------------')

        agent_data = sim.take_sample(i_traj)

        if "demo_images" in agent_data:
            agent_data.pop("demo_images")

        stat_arrays = OrderedDict()
        for key in agent_data.keys():
            if key not in stats_lists:
                stats_lists[key] = []
            stats_lists[key].append(agent_data[key])
            stat_arrays[key] = np.array(stats_lists[key])

        i_traj +=1 #increment trajectories every step!

        pickle.dump(stat_arrays, open(final_dist_pkl_file, 'wb'))
        if isinstance(sim.agent, BenchmarkAgent):
            write_scores(conf, result_file, stat_arrays, i_traj)


if __name__ == '__main__':
    run_trajectories()
