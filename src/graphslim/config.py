'''Configuration'''
import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import json
import logging

import click
from pprint import pformat
import graphslim
from graphslim.utils import seed_everything, f1_macro, accuracy, roc_auc

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        # Use pprint's pformat to print the dictionary in a pretty manner
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


def update_from_dict(obj, updates):
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)


# fix setting here
def setting_config(args):
    representative_r = {
        'cora': 0.5,
        'citeseer': 0.5,
        'pubmed': 0.5,
        'flickr': 0.01,
        'reddit': 0.001,
        'ogbn-arxiv': 0.01,
        'yelp': 0.001,
        'amazon': 0.002,
        'pubmed': 0.1
    }
    if args.reduction_rate == -1:
        args.reduction_rate = representative_r[args.dataset]
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'dblp']:
        args.setting = 'trans'
    if args.dataset in ['flickr', 'reddit', 'amazon','yelp']:
        args.setting = 'ind'
    # args.pre_norm = True
    # args.metric = f1_macro if args.dataset in ['yelp', 'amazon'] else accuracy
    args.metric = f1_macro if args.dataset in ['yelp', 'amazon'] else accuracy
    
    
    # args.metric = 'accuracy'
    args.run_inter_eval = 3
    args.eval_interval = max(args.epochs // 10, 1)
    # if args.method not in ['gcsntk']:
    #     args.eval_interval = max(args.epochs // 10, 1)
    args.checkpoints = list(range(-1, args.epochs + 1, args.eval_interval))
    args.eval_epochs = 300
    return args


# recommend hyperparameters here
def method_config(args):
    # try:
    #     # print(os.path.dirname(graphslim.__file__))
    #     conf_dt = json.load(
    #         open(f"{os.path.join(os.path.dirname(graphslim.__file__), 'configs', args.method, args.dataset,args.reduction_rate)}.json"))
    #     update_from_dict(args, conf_dt)

    try:
        # print(os.path.dirname(graphslim.__file__))
        conf_dt = json.load(
            open(f"{os.path.join(os.path.dirname(graphslim.__file__), 'configs', args.method, (args.dataset).lower().replace('-', '').replace('_', ''))}.json"))
        update_from_dict(args, conf_dt)
   
    except:
        print('No config file found or error in json format, please use method_config(args)')
    if args.method in ['msgc']:
        args.batch_adj = 16
        # add temporary changes here
        # do not modify the config json

    return args


@click.command()
# HyDRO
@click.option('--momentum', default=0.01, show_default=True)
@click.option('--weight_decay_hyper', default=5e-4, show_default=True)
@click.option('--beta', default=0.1, type=float, show_default=True,
              help='HyDRO regularization coefficient')
@click.option('--curvature', default=-0.1, type=float, show_default=True,
              help='sectional curvature kappa; Geoopt receives its magnitude')
@click.option(
    '--spectral-objective/--no-spectral-objective',
    default=True,
    show_default=True,
    help=(
        'Backpropagate the spectral-gap term from Algorithm 1. '
        'The public code accidentally added this term after backward().'
    ),
)
@click.option('--metric', default='accuracy', show_default=True)

@click.option('--dataset', '-D', default='cora', show_default=True)
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', type=click.Choice(['trans', 'ind']), show_default=True,
              help='transductive or inductive setting')
@click.option('--split', default='fixed', show_default=True,
              help='only support public split now, do not change it')  # 'fixed', 'random', 'few'
@click.option('--run_reduction', default=3, show_default=True, help='repeat times of reduction')
@click.option('--run_eval', default=10, show_default=True, help='repeat times of final evaluations')
@click.option('--run_inter_eval', default=5, show_default=True, help='repeat times of intermediate evaluations')
@click.option('--eval_interval', default=100, show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--eval_epochs', '--ee', default=300, show_default=True)
@click.option('--eval_model', '--em', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT', 'SGFormer']
              ), show_default=True)
@click.option('--final_eval_model', '--fem', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT', 'SGFormer']
              ), show_default=True)
@click.option('--condense_model', default='SGC',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
              ), show_default=True)
@click.option('--epochs', '-E', default=1000, show_default=True, help='number of reduction epochs')
# @click.option('--valid_result', '--vr', default=0.0, show_default=True)
# @click.option('--patience', '-P', default=20, show_default=True)  # only for msgc
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=0.0, show_default=True)
# @click.option('--normalize_features', is_flag=True, show_default=True)
@click.option('--pre_norm', default=True, show_default=True,
              help='pre-normalize features, forced true for arxiv, flickr and reddit')
@click.option('--outer_loop', default=10, show_default=True)
@click.option('--inner_loop', default=1, show_default=True)
@click.option('--reduction_rate', '-R', default=1.0, show_default=True,
              help='-1 means use representative reduction rate; reduction rate of training set, defined as (number of nodes in small graph)/(number of nodes in original graph)')
@click.option('--seed', '-S', default=1, help='Random seed', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers of condensed model', show_default=True)
@click.option('--verbose', '-V', is_flag=True, show_default=True)
@click.option('--soft_label', default=0, show_default=True)
@click.option('--init', default='random', help='features initialization methods',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random']
              ), show_default=True)
# @click.option('--method', '-M', default='kcenter',
#               type=click.Choice(
#                   ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
#                    'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
#                    'gcond', 'doscond', 'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom','cadm',
#                    'cent_d', 'cent_p', 'kcenter', 'herding', 'random',
#                    'random_edge'
#                    ]), show_default=True)

@click.option('--eval_wd', '--ewd', default=0.0, show_default=True)
@click.option('--eval_loss', '--eloss', default='CE',
                type=click.Choice(
                  ['CE', 'KLD','MSE']
              ), show_default=True)
@click.option('--method', '-M', default='kcenter', show_default=True)
@click.option('--activation', default='relu', help='activation function when do NAS',
              type=click.Choice(
                  ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
              ), show_default=True)
@click.option('--attack', '-A', default=None, help='corruption method',
              type=click.Choice(
                  ['random_adj', 'metattack', 'random_feat']
              ), show_default=True)
@click.option('--coarsen_strategy', '--cs', default='greedy', help='for edge contraction method',
              type=click.Choice(
                  ['optimal', 'greedy']
              ), show_default=True)
@click.option('--ptb_r', '-P', default=0.25, show_default=True, help='perturbation rate for corruptions')
@click.option('--agg', is_flag=True, show_default=True, help='use aggregation for coreset methods')
@click.option('--multi_label', is_flag=True, show_default=True, help='use aggregation for coreset methods')
@click.option('--dis_metric', default='ours', show_default=True,
              help='distance metric for all condensation methods,ours means metric used in GCond paper')
@click.option('--lr_adj', default=1e-4, show_default=True)
@click.option('--lr_feat', default=1e-4, show_default=True)
@click.option('--optim', default="Adam", show_default=True)
@click.option('--threshold', default=0.0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--no_buff', is_flag=True, show_default=True,
              help='skip the buffer generation and use existing in geom,sfgc')
@click.option('--batch_adj', default=1, show_default=True, help='batch size for msgc')
# model specific args
@click.option('--alpha', default=0.1, help='for appnp', show_default=True)
@click.option('--mx_size', default=100, help='for gcsntk methods, avoid SVD error', show_default=True)
@click.option('--ts', default=4, help='for tspanner', show_default=True)
@click.option(
    '--save_path',
    '--sp',
    default=os.path.join(_REPO_ROOT, 'checkpoints'),
    show_default=True,
    help='save path for synthetic graph',
)
@click.option(
    '--load_path',
    '--lp',
    default=os.path.join(_REPO_ROOT, 'data'),
    show_default=True,
    help='dataset root',
)
@click.option('--lp_hidden', default=128, type=int, show_default=True,
              help='hidden units in the paper link-prediction evaluator')
@click.option('--lp_lr', default=0.001, type=float, show_default=True,
              help='learning rate in the paper link-prediction evaluator')
@click.option('--lp_epochs', default=100, type=int, show_default=True,
              help='training epochs in the paper link-prediction evaluator')
@click.option('--lp_weight_decay', default=0.0, type=float, show_default=True)
@click.option('--lp_neg_ratio', default=1.0, type=float, show_default=True,
              help='negative-to-positive ratio for link prediction')
@click.option('--lp_val_interval', default=10, type=int, show_default=True)
@click.option('--lp_decision_threshold', default=0.5, type=float, show_default=True)
@click.option('--lp_f1_average', default='macro',
              type=click.Choice(['binary', 'macro']), show_default=True)
@click.option('--lp_syn_thresholds', default='q55,q65,q75,q85,q95',
              show_default=True,
              help='validation-only thresholds for the dense synthetic adjacency')
@click.option(
    '--paper_lapt_source_path',
    default=None,
    type=click.Path(file_okay=False, path_type=str),
    help=(
        'checkpoint root containing the paper-protocol HyDRO graph used '
        'as the LAPT source'
    ),
)
@click.option(
    '--paper_lapt_lambdas',
    default='0,0.1,0.2,0.3,0.5,0.75,1.0',
    show_default=True,
    help='validation-only mixture weights for HyDRO and the LAPT target',
)
@click.option('--paper_lapt_validation_runs', default=3, type=int,
              show_default=True)
@click.option('--paper_lapt_proxy_runs', default=3, type=int,
              show_default=True,
              help='training-query GCN proxy model seeds per topology candidate')
@click.option('--paper_lapt_proxy_epochs', default=100, type=int,
              show_default=True,
              help='training epochs for the leakage-safe GCN proxy')
@click.option('--paper_lapt_proxy_val_interval', default=100, type=int,
              show_default=True,
              help='training-query scoring interval for the GCN proxy')
@click.option('--paper_lapt_proxy_risk_weight', default=0.5, type=float,
              show_default=True,
              help='standard-deviation penalty in proxy candidate selection')
@click.option('--paper_lapt_proxy_thresholds', default='q55,q65,q75,q85,q95',
              show_default=True,
              help='synthetic adjacency thresholds used by the GCN proxy')
@click.option('--paper_lapt_proxy_split_seeds', default='11,23,37',
              show_default=True,
              help='support/query resampling seeds within the fixed training partition')
@click.option('--paper_lapt_proxy_gate_z', default=1.0, type=float,
              show_default=True,
              help='cross-view standard-deviation multiplier for the paired gain lower bound')
@click.option('--paper_lapt_proxy_min_gain', default=0.001, type=float,
              show_default=True,
              help='minimum paired proxy risk gain required by the fallback gate')
@click.option('--paper_lapt_proxy_min_win_rate', default=0.67, type=float,
              show_default=True,
              help='minimum fraction of query views won by a LAPT candidate')
@click.option('--paper_lapt_split_seed', default=1, type=int,
              show_default=True)
@click.option('--paper_lapt_validation_ratio', default=0.10, type=float,
              show_default=True)
@click.option('--paper_lapt_test_ratio', default=0.20, type=float,
              show_default=True)
@click.option('--paper_lapt_query_ratio', default=0.10, type=float,
              show_default=True)
@click.option('--paper_lapt_hard_negative_ratio', default=0.0, type=float,
              show_default=True)
@click.option('--paper_lapt_teacher_hidden', default=32, type=int,
              show_default=True)
@click.option('--paper_lapt_teacher_pair_limit', default=8192, type=int,
              show_default=True)
@click.option('--paper_lapt_teacher_steps', default=200, type=int,
              show_default=True)
@click.option('--paper_lapt_teacher_batch_size', default=2048, type=int,
              show_default=True)
@click.option('--paper_lapt_teacher_lr', default=0.01, type=float,
              show_default=True)
@click.option('--paper_lapt_teacher_batch_nodes', default=256, type=int,
              show_default=True)
@click.option(
    '--paper_lapt_use_structure_context/--no-paper_lapt_use_structure_context',
    default=False,
    show_default=True,
    help=(
        'align support-graph and condensed-graph propagated feature contexts '
        'during prototype assignment'
    ),
)
@click.option(
    '--paper_lapt_structure_blends',
    default='0,0.25,0.5,0.75,1.0',
    show_default=True,
    help=(
        'training-query-selected blends of raw feature and support-graph '
        'structural-context similarities; must include 0'
    ),
)
@click.option('--paper_lapt_structure_steps', default=2, type=int,
              show_default=True)
@click.option('--paper_lapt_assignment_blends',
              default='0.25,0.5,0.75,1.0', show_default=True)
@click.option('--paper_lapt_prototype_temperature', default=0.1, type=float,
              show_default=True)
@click.option('--paper_lapt_sinkhorn_iter', default=20, type=int,
              show_default=True)
@click.option(
    '--paper_lapt_class_aware/--no-paper_lapt_class_aware',
    default=True,
    show_default=True,
)
@click.option(
    '--paper_lapt_use_teacher/--no-paper_lapt_use_teacher',
    default=True,
    show_default=True,
    help='use the link teacher in prototype assignment',
)
@click.option(
    '--paper_lapt_use_rank_calibration/--no-paper_lapt_use_rank_calibration',
    default=True,
    show_default=True,
    help='rank-match the LAPT target to the HyDRO weight distribution',
)
@click.option(
    '--paper_lapt_rank_blends',
    default='0,0.5,1.0',
    show_default=True,
    help=(
        'validation-only strengths for interpolating the raw LAPT target '
        '(0) and the fully rank-calibrated target (1)'
    ),
)
@click.option(
    '--paper_lapt_selection_tolerance',
    default=0.001,
    type=float,
    show_default=True,
    help=(
        'absolute validation-F1 tolerance used to prefer a smaller rank '
        'strength and then a smaller topology mixture weight'
    ),
)
@click.option(
    '--paper_lapt_selection_mode',
    default='validation',
    type=click.Choice(['validation', 'fixed']),
    show_default=True,
    help='select lambda on validation data or use a pre-registered fixed value',
)
@click.option('--paper_lapt_fixed_lambda', default=0.3, type=float,
              show_default=True)
@click.option('--paper_lapt_fixed_rank_blend', default=0.0, type=float,
              show_default=True)
@click.option('--paper_lapt_query_calibration_fraction', default=0.5,
              type=float, show_default=True)
@click.option('--paper_lapt_query_target_weight', default=0.25, type=float,
              show_default=True)
@click.option('--paper_lapt_target_budget_fraction', default=0.2, type=float,
              show_default=True)
@click.option('--paper_lapt_target_temperature', default=0.5, type=float,
              show_default=True)
@click.option('--paper_lapt_min_teacher_auc', default=0.55, type=float,
              show_default=True)
@click.option('--paper_lapt_min_target_query_auc', default=0.52, type=float,
              show_default=True)
@click.option('--paper_lapt_min_target_query_gap', default=0.002, type=float,
              show_default=True)
@click.option(
    '--scltc_source_paths',
    default=None,
    help=(
        'comma-separated Step23-compatible HyDRO checkpoint roots, one for '
        'each condensation seed'
    ),
)
@click.option(
    '--scltc_source_seeds',
    default='1,2,3',
    show_default=True,
    help='condensation seeds corresponding to scltc_source_paths',
)
@click.option(
    '--scltc_lambdas',
    default='0,0.1,0.2,0.3,0.5,0.75,1.0',
    show_default=True,
    help='shared topology-mixture candidates selected only on validation data',
)
@click.option(
    '--scltc_rank_blends',
    default='0,0.5,1.0',
    show_default=True,
    help='shared raw/rank target blends selected only on validation data',
)
@click.option('--scltc_risk_weight', default=0.5, type=float,
              show_default=True,
              help='penalty on validation F1 variation across condensation seeds')
@click.option('--scltc_canonical_seed', default=1, type=int,
              show_default=True)
@click.option('--scltc_validation_runs', default=3, type=int,
              show_default=True)
@click.option('--scltc_selection_tolerance', default=0.001, type=float,
              show_default=True,
              help='absolute validation-risk tolerance; ties prefer weaker changes')
@click.option('--scltc_eval_seed', default=1, type=int,
              show_default=True,
              help='fixed LP split/model seed and saved checkpoint filename seed')
@click.option('--scltc_local_anchor_seed', default=3, type=int,
              show_default=True,
              help='condensation seed used as the local-disagreement anchor')
@click.option('--scltc_local_fractions', default='0.02,0.05,0.1', type=str,
              show_default=True,
              help='upper-triangle fractions considered for local rank correction')
@click.option('--scltc_local_lambdas', default='0,0.25,0.5,1.0', type=str,
              show_default=True,
              help='local rank-target mixture strengths; must include zero')
@click.option('--scltc_local_agreement_margin', default=0.02, type=float,
              show_default=True,
              help='minimum peer-versus-anchor rank gap for directional agreement')
@click.option('--scltc_router_raw_paths', default=None, type=str,
              help='comma-separated raw HyDRO roots for validation routing')
@click.option('--scltc_router_rctc_paths', default=None, type=str,
              help='comma-separated RCTC roots for validation routing')
@click.option('--scltc_router_lapt_paths', default=None, type=str,
              help='comma-separated adaptive-LAPT roots for validation routing')
@click.option('--scltc_fusion_rctc_paths', default=None, type=str,
              help='comma-separated RCTC roots for complementary topology fusion')
@click.option('--scltc_fusion_lapt_paths', default=None, type=str,
              help='comma-separated adaptive-LAPT roots for complementary topology fusion')
@click.option('--scltc_fusion_blends', default='0,0.25,0.5,0.75,1.0', type=str,
              show_default=True,
              help='RCTC-to-LAPT convex adjacency blends selected on validation data')
@click.option('--scltc_rank_fusion_rctc_paths', default=None, type=str,
              help='comma-separated RCTC roots for nonlinear paired rank fusion')
@click.option('--scltc_rank_fusion_lapt_paths', default=None, type=str,
              help='comma-separated adaptive-LAPT roots for nonlinear paired rank fusion')
@click.option('--scltc_rank_fusion_modes',
              default='mean_rank,rank_union,rank_intersection', type=str,
              show_default=True,
              help='paired rank operators considered by validation selection')
@click.option('--scltc_rank_fusion_references', default='rctc,lapt', type=str,
              show_default=True,
              help='endpoint edge-weight distributions restored after rank fusion')
@click.option('--scltc_teacher_gate_rctc_paths', default=None, type=str,
              help='comma-separated RCTC roots for teacher-gated local ranking')
@click.option('--scltc_teacher_gate_lapt_paths', default=None, type=str,
              help='comma-separated adaptive-LAPT roots for teacher-gated local ranking')
@click.option('--scltc_teacher_gate_fractions', default='0.02,0.05,0.1,0.2',
              type=str, show_default=True,
              help='fractions of most-disputed pairs exposed to the teacher gate')
@click.option('--scltc_teacher_gate_references', default='rctc,lapt', type=str,
              show_default=True,
              help='endpoint weight multisets used by teacher-gated candidates')
@click.option('--scltc_teacher_gate_pair_batch_size', default=8192, type=int,
              show_default=True,
              help='synthetic-pair batch size for training-only teacher scoring')
@click.option('--condensed_selection_metric', default='nc',
              type=click.Choice(['nc', 'lp']), show_default=True,
              help='validation metric used to select a condensed checkpoint')
@click.option('--eval_whole', '-W', is_flag=True, show_default=True, help='if run on whole graph')
@click.option('--with_structure', default=1, show_default=True, help='if synthesizing structure')
# ======================================simgc====================================== #
@click.option('--feat_alpha', default=10, show_default=True, help='feature loss weight')
@click.option('--smoothness_alpha', default=0.1, show_default=True, help='smoothness loss weight')
#==================gdem=====#
@click.option('--eigen_k', default=60, show_default=True, help='number of eigenvalues')
@click.option('--ratio', default=0.8, show_default=True, help='eigenvalue loss weight')
@click.option('--lr_eigenvec', default=0.01, show_default=True, help='eigenvalue loss weight')
@click.option('--gamma', default=0.5, show_default=True, help='eigenvalue loss weight')
@click.pass_context
def cli(ctx, **kwargs):
    args = dict2obj(kwargs)
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        args.device = f'cuda:0'
    else:
        # if gpu_id=-1, use cpu
        args.device = 'cpu'
    path = args.save_path
    # for benchmark, we need unified settings and reduce flexibility of args
    args = method_config(args)
    # setting_config has higher priority than methods_config
    args = setting_config(args)
    for key, value in ctx.params.items():
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE:
            setattr(args, key, value)
    if (
        ctx.get_parameter_source('eval_interval')
        != click.core.ParameterSource.COMMANDLINE
    ):
        args.eval_interval = max(args.epochs // 10, 1)
    args.checkpoints = list(
        range(-1, args.epochs + 1, args.eval_interval)
    )
    if not os.path.exists(f'{path}/logs/{args.method}'):
        try:
            os.makedirs(f'{path}/logs/{args.method}')
        except:
            print(f'{path}/logs/{args.method} exists!')
    logging.basicConfig(filename=f'{path}/logs/{args.method}/{args.dataset}_{args.reduction_rate}.log',
                        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args.logger = logging.getLogger(__name__)
    args.logger.addHandler(logging.StreamHandler())
    args.logger.info(args)
    return args


def get_args():
    return cli(standalone_mode=False)


if __name__ == '__main__':
    cli()
