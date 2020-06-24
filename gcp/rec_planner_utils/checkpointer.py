import glob
import os
import pipes
import sys

import numpy as np
import torch
from blox.basic_types import str2int

class NoCheckpointsException(Exception):
    pass


class CheckpointHandler:
    @staticmethod
    def get_ckpt_name(epoch):
        return 'weights_ep{}.pth'.format(epoch)
    
    @staticmethod
    def get_epochs(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            print("Warning: No checkpoints found at {}!".format(path))
            raise NoCheckpointsException
        processed_names = [file.split('/')[-1].replace('weights_ep', '').replace('.pth', '')
                           for file in checkpoint_names]
        epochs = list(filter(lambda x: x is not None, [str2int(name) for name in processed_names]))
        return epochs
    
    @staticmethod
    def get_resume_ckpt_file(resume, path):
        print("Loading from: {}".format(path))
        if resume == 'latest':
            max_epoch = np.max(CheckpointHandler.get_epochs(path))
            resume_file = CheckpointHandler.get_ckpt_name(max_epoch)
        elif str2int(resume) is not None:
            resume_file = CheckpointHandler.get_ckpt_name(resume)
        elif '.pth' not in resume:
            resume_file = resume + '.pth'
        else:
            resume_file = resume
        return os.path.join(path, resume_file)

    @staticmethod
    def load_weights(weights_file, model, load_step_and_opt=False, optimizer=None, dataset_length=None, strict=True,
                     submodule_name=None):
        success = False
        if os.path.isfile(weights_file):
            print(("=> loading checkpoint '{}'".format(weights_file)))
            checkpoint = torch.load(weights_file, map_location=model.device)
            model.load_state_dict(CheckpointHandler.filter(checkpoint['state_dict'], submodule_name), strict=strict)
            if load_step_and_opt:
                start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint['global_step']
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except (RuntimeError, ValueError) as e:
                    if not strict:
                        print("Could not load optimizer params because of changes in the network + non-strict loading")
                        pass
                    else:
                        raise e
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights_file, checkpoint['epoch'])))
            success = True
        else:
            # print(("=> no checkpoint found at '{}'".format(weights_file)))
            # start_epoch = 0
            raise ValueError("Could not find checkpoint file in {}!".format(weights_file))
        
        if load_step_and_opt:
            return global_step, start_epoch, success
        else:
            return success

    
    @staticmethod
    def hack_to_fix_checkpoints():
        import torch
        fold = '../experiments/prediction/rec_planner/nav2d/single_wall/soft_fixed_baseline/kl1e-2'
        keywords_remove = ['running_var', 'running_mean', 'num_batches_tracked']
        
        ckpt = torch.load(fold + '/weights_ep17.pth')
        state = dict(ckpt['state_dict'])
        new_state = {}
    
        def remove_keyword(keyword):
            pop_list = ([key for key in state if keyword in key]);
            [state.pop(key) for key in pop_list]
            
        [remove_keyword(keyword) for keyword in keywords_remove]
    
        def fn(key): new_state[key.replace('batchnorm', 'norm')] = state[key]
    
        [fn(key) for key in state if 'batchnorm' in key]
        remove_keyword('batchnorm')
        
        new_state.update(state)
        ckpt['state_dict'] = new_state
        torch.save(ckpt, fold + '/weights_norun.pth')

    @staticmethod
    def another_hack_to_fix_checkpoints():
        import torch
        fold = '../experiments/prediction/rec_planner/nav2d/single_wall/soft_fixed_baseline/kl1e-2'
        keywords_remove = ['running_var', 'running_mean', 'num_batches_tracked']

        ckpt = torch.load(fold + '/weights_ep17.pth')
        state = dict(ckpt['state_dict'])
        new_state = {}

        def remove_keyword(keyword):
            pop_list = ([key for key in state if keyword in key]);
            [state.pop(key) for key in pop_list]

        [remove_keyword(keyword) for keyword in keywords_remove]

        new_state.update(state)
        ckpt['state_dict'] = new_state
        torch.save(ckpt, fold + '/weights_norun.pth')
        
        
    @staticmethod
    def rename_parameters(dict, old, new):
        """ Renames parameters in the network by finding parameters that contain 'old' and replacing 'old' with 'new'
        """
        replacements = [key for key in dict if old in key]
        
        for key in replacements:
            dict[key.replace(old, new)] = dict.pop(key)

    @staticmethod
    def filter(state_dict, submodule_key):
        """Filters state_dict by searching for key and removing all other elements + key-prefix (to lead submodule)."""
        if submodule_key is None: return state_dict

        new_dict = {}
        for key in state_dict:
            if key.startswith(submodule_key):
                new_dict[key[len(submodule_key)+1:]] = state_dict[key]
        if not new_dict:
            raise ValueError("Did not find submodule {} in checkpoint!".format(submodule_key))
        return new_dict


def get_config_path(path):
    conf_names = glob.glob(os.path.abspath(path) + "/*.py")
    if len(conf_names) == 0:
        raise ValueError("No configuration files found at {}!".format(path))

    # The standard conf
    if 'conf.py' in map(lambda x: x.split('/')[-1], conf_names):
        return os.path.join(path, 'conf.py')

    # Get latest conf
    arrays = [np.array(file.split('__')[-1].replace('_', '-').replace('.py', '').split('-'),
                       dtype=float) for file in filter(lambda x: '__' in x, conf_names)]
    # Converts arrays representing time to values that can be compared
    values = np.array(list([ar[5] + 100 * ar[4] + (100 ** 2) * ar[3] + (100 ** 3) * ar[2]
                            + (100 ** 4) * ar[1] + (100 ** 5) * 10 * ar[0]
                            for ar in arrays]))
    conf_ind = np.argmax(values)
    return conf_names[conf_ind]
    
    
def save_git(base_dir):
  # save code revision
  print('Save git commit and diff to {}/git.txt'.format(base_dir))
  cmds = ["echo `git rev-parse HEAD` > {}".format(
    os.path.join(base_dir, 'git.txt')),
    "git diff >> {}".format(
      os.path.join(base_dir, 'git.txt'))]
  print(cmds)
  os.system("\n".join(cmds))


def save_cmd(base_dir):
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(os.path.join(base_dir, "cmd.txt"), "w") as f:
    f.write(train_cmd)

                

