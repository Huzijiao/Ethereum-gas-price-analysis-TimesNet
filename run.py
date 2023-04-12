import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import yaml

# 可以改成自己的yaml路径
cfg_path = './cfg/test14.yaml'

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    f = open(cfg_path, encoding='utf-8')
    data = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()

    parser = argparse.ArgumentParser(description='TimesNet')
    # basic config
    parser.add_argument('--task_name', type=str,  default=data['task_name'],
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=data['is_training'], help='status')
    parser.add_argument('--model_id', type=str,  default=data['model_id'], help='model id')
    parser.add_argument('--model', type=str,  default=data['model'],
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str,  default=data['data'], help='dataset type')
    parser.add_argument('--root_path', type=str, default=data['root_path'], help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=data['data_path'], help='data file')
    parser.add_argument('--features', type=str, default=data['features'],
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default=data['target'], help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default=data['freq'],
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default=data['checkpoints'], help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=data['seq_len'], help='input sequence length')
    parser.add_argument('--label_len', type=int, default=data['label_len'], help='start token length')
    parser.add_argument('--pred_len', type=int, default=data['pred_len'], help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default=data['seasonal_patterns'], help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=data['mask_rate'], help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=data['anomaly_ratio'], help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=data['top_k'], help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=data['num_kernels'], help='for Inception')
    parser.add_argument('--enc_in', type=int, default=data['enc_in'], help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=data['dec_in'], help='decoder input size')
    parser.add_argument('--c_out', type=int, default=data['c_out'], help='output size')
    parser.add_argument('--d_model', type=int, default=data['d_model'], help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=data['n_heads'], help='num of heads')
    parser.add_argument('--e_layers', type=int, default=data['e_layers'], help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=data['d_layers'], help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=data['d_ff'], help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=data['moving_avg'], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=data['factor'], help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=data['distil'])
    parser.add_argument('--dropout', type=float, default=data['dropout'], help='dropout')
    parser.add_argument('--embed', type=str, default=data['embed'],
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default=data['activation'], help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=data['num_workers'], help='data loader num workers')
    parser.add_argument('--itr', type=int, default=data['itr'], help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=data['train_epochs'], help='train epochs')
    parser.add_argument('--batch_size', type=int, default=data['batch_size'], help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=data['patience'], help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=data['learning_rate'], help='optimizer learning rate')
    parser.add_argument('--des', type=str, default=data['des'], help='exp description')
    parser.add_argument('--loss', type=str, default=data['loss'], help='loss function')
    parser.add_argument('--lradj', type=str, default=data['lradj'], help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=data['use_amp'])

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=data['use_gpu'], help='use gpu')
    parser.add_argument('--gpu', type=int, default=data['gpu'], help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=data['use_multi_gpu'])
    parser.add_argument('--devices', type=str, default=data['devices'], help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=data['p_hidden_dims'],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=data['p_hidden_layers'], help='number of hidden layers in projector')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    key = data.keys()
    print("config: ")
    print("-----------------------------------------------------")
    for item in key:
        print("{} : {}".format(item, data[item]))

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
