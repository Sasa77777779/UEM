import os
import yaml
import time


cfg = {}


cfg['model_name'] = 'UEM_model'
cfg['dataset_name'] = 'tvr'
cfg['seed'] = 9527

cfg['root'] = '/UEM/'
cfg['data_root'] = '/data/'

cfg['collection'] = 'tvr'

cfg['model_root'] = os.path.join(cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'], time.strftime("%Y_%m_%d_%H_%M_%S"))
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# extra
cfg['sft_factor'] = 0.09


# dataset
cfg['num_workers'] = 32
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 64


# opt
cfg['lr'] = 0.0003
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False

cfg['lambda'] = 0.02
cfg['threshold'] = 0.92


# eval
cfg['eval_query_bsz'] = 30
cfg['eval_context_bsz'] = 100


# model
# max word number
cfg['max_desc_l'] = 30
# max frame number
cfg['max_ctx_l'] = 128
cfg['sub_feat_size'] = 768

# text feature dimension
cfg['q_feat_size'] = 512
# frame feature dimension
cfg['visual_feat_dim'] = 512

cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']


if not os.path.exists(cfg['model_root']):
    os.makedirs(cfg['model_root'], exist_ok=True)
if not os.path.exists(cfg['ckpt_path']):
    os.makedirs(cfg['ckpt_path'], exist_ok=True)


def get_cfg_defaults():
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg