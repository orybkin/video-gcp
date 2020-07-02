from blox import AttrDict


def get_default_gcp_hyperparameters():
    
    # Params that actually should be elsewhere
    default_dict = AttrDict({
        'randomize_length': False,
        'randomize_start': False,
    })
    
    # Network size
    default_dict.update({
        'ngf': 4,  # number of feature maps in shallowest level
        'nz_enc': 32,  # number of dimensions in encoder-latent space
        'nz_vae': 32,  # number of dimensions in vae-latent space
        'nz_vae2': 256,  # number of dimensions in 2nd level vae-latent space (if used)
        'nz_mid': 32,  # number of dimensions for internal feature spaces
        'nz_mid_lstm': 32,
        'n_lstm_layers': 1,
        'n_processing_layers': 3,  # Number of layers in MLPs
        'conv_inf_enc_kernel_size': 3,  # kernel size of convolutional inference encoder
        'conv_inf_enc_layers': 1,  # number of layers in convolutional inference encoder
        'n_attention_heads': 1,  # number of attention heads (needs to divide nz_enc evenly)
        'n_attention_layers': 1,  # number of layers in attention network
        'nz_attn_key': 32,  # dimensionality of the attention key
        'init_mlp_layers': 3,  # number of layers in the LSTM initialization MLP (if used)
        'init_mlp_mid_sz': 32,  # size of hidden layers inside LSTM initialization MLP (if used)
        'n_conv_layers': None,  # Number of conv layers. Can be of format 'n-<int>' for any int for relative spec
    })
    
    # Network params
    default_dict.update(AttrDict(
        action_activation=None,
        device=None,
        context_every_step=True,
    ))
    
    # Loss weights
    default_dict.update({
        'kl_weight': 1.,
        'kl_weight_burn_in': None,
        'entropy_weight': .0,
        'length_pred_weight': 1.,
        'dense_img_rec_weight': 1.,
        'dense_action_rec_weight': 1.,
        'free_nats': 0,
    })
    
    # Architecture params
    default_dict.update({
        'use_skips': True,  # only works with conv encoder/decoder
        'skips_stride': 2,
        'add_weighted_pixel_copy': False,  # if True, adds pixel copying stream for decoder
        'pixel_shift_decoder': False,
        'skip_from_parents': False,  # If True, parents are added to the pixel copy/shift sources
        'seq_enc': 'none',  # Manner of sequence encoding. ['none', 'conv', 'lstm']
        'regress_actions': False,  # whether or not to regress actions
        'learn_attn_temp': True,  # if True, makes attention temperature a trainable variable
        'attention_temperature': 1.0,  # temperature param used in attention softmax
        'attach_inv_mdl': False,  # if True, attaches an inverse model to output that computes actions
        'attach_cost_mdl': False,  # if True, attaches a cost function MLP that estimates cost from pairs of states
        'run_cost_mdl': True,   # if False, does not run cost model (but might still build it
        'attach_state_regressor': False,    # if True, attaches network that regresses states from pre-decoding-latents
        'action_conditioned_pred': False,  # if True, conditions prediction on actions
        'learn_beta': True,  # if True, learns beta
        'initial_sigma': 1.0,  # if True, learns beta
        'separate_cnn_start_goal_encoder': False,   # if True, builds separate conv encoder for start/goal image
        'decoder_distribution': 'gaussian'  # [gaussian, categorical]
    })

    # RNN params
    default_dict.update({
        'use_conv_lstm': False,
    })
    
    # Variational inference parameters
    default_dict.update(AttrDict(
        prior_type='learned',  # type of prior to be used ['fixed', 'learned']
        var_inf='standard',  # type of variation inference ['standard', '2layer', 'deterministic']
    ))
    
    # RecPlan params
    default_dict.update({
        'hierarchy_levels': 3,  # number of levels in the subgoal tree
        
        'one_hot_attn_time_cond': False,  # if True, conditions attention on one-hot time step index
        'attentive_inference': False,  # if True, forces attention to single matching frame
        'non_goal_conditioned': False,  # if True, does not condition prediction on goal frame
        
        'tree_lstm': '',  # ['', 'sum' or 'linear']
        'lstm_init': 'zero',  # defines how treeLSTM is initialized, ['zero', 'mlp'], #, 'warmup']
        
        'matching_temp': 1.0,  # temperature used in TAP-style matching softmax
        'matching_temp_tenthlife': -1,
        'matching_temp_min': 1e-3,
        'matching_type': 'latent',  # evidence binding procedure
        # ['image', 'latent', 'fraction', 'balanced', 'tap']
        'leaves_bias': 0.0,
        'top_bias': 1.0,
        'n_top_bias_nodes': 1,
        'supervise_match_weight': 0.0,
        
        'regress_index': False,
        'regress_length': False,
        
        'inv_mdl_params': {},  # params for the inverse model, if attached
        'train_inv_mdl_full_seq': False,  # if True, omits sampling for inverse model and trains on full seq

        'cost_mdl_params': {},   # cost model parameters

        'act_cond_inference': False,  # if True, conditions inference on actions

        'train_on_action_seqs': False,  # if True, trains the predictive network on action sequences

        'learned_pruning_threshold': 0.5,   # confidence thresh for learned pruning network after which node gets pruned
        'untied_layers': False,
        'supervised_decoder': False,
        'states_inference': False,
    })
    
    # Outdated GCP params
    default_dict.update({
        'dense_rec_type': 'none',  # ['none', 'discrete', 'softmax', 'linear', 'node_prob', action_prob].
        'one_step_planner': 'discrete',  # ['discrete', 'continuous', 'sh_pred']. Always 'sh_pred' for HEDGE models
        'mask_inf_attention': False,  # if True, masks out inference attention outside the current subsegment
        'binding': 'frames',  # Matching loss form ['loss', 'frames', 'exp', 'lf']. Always 'loss'.
    })
    
    # Matching params
    default_dict.update(AttrDict(
        learn_matching_temp=True,  # If true, the matching temperature is learned
    ))
    
    # Logging params
    default_dict.update(AttrDict(
        dump_encodings='',  # Specifies the directory where to dump the encodings
        dump_encodings_inv_model='',  # Specifies the directory where to dump the encodings
        log_states_2d=False,  # if True, logs 2D plot of first two state dimensions
        log_cartgripper=False,  # if True, logs sawyer from states
        data_dir='',   # necessary for sawyer logging
    ))

    # Hyperparameters that shouldn't exist
    default_dict.update(AttrDict(
        log_d2b_3x3maze=0,  # Specifies the directory where to dump the encodings
    ))
    
    
    return default_dict
