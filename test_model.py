from utils.json_config import JsonConfig
import torch

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

import numpy as np
import argparse
import os

from utils.dataloader import *
from utils.loss_function import *
from model.intentgcn import IntentGCN, Multi_LaterFusion_PD

def get_param_num(model, require_grad_param=True):
    if require_grad_param:
        return sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    else:
        return sum(param.numel() for param in model.parameters())
    
def result_ana(labels, preds):
    result = {}
    result['accuracy'] = accuracy_score(labels, preds) * 100
    result['precision'] = precision_score(labels, preds, average='binary') * 100
    result['recall'] = recall_score(labels, preds, average='binary') * 100
    result['fscore'] = f1_score(labels, preds, average='binary') * 100
    result['auc'] = roc_auc_score(labels, preds) * 100
    result['labels'] = labels
    result['preds'] = preds
    print('*' * 20)
    print('accuracy: ', result['accuracy'])
    print('precision: ', result['precision'])
    print('recall: ', result['recall'])
    print('fscore: ', result['fscore'])
    print('auc: ', result['auc'])
    print('*' * 20)
    return result

def main(args):

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:"+str(args.Data.device_index))
        else:
            device = torch.device("cuda:0")
        print('[Info] Device: {}'.format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print('[Info] Device: {}'.format(device))

    train_set, _, _, _, _, test_loader = get_JAAD_dataloader(args.Data.path, args.Data.features, args.Optim.batch_size)
    input_shape = train_set.get_shape()

    print(f'Input shape: {input_shape}')

    if args.Model.model == 'intentgcn':

        args.GCN_paras = {k:v for k,v in args.GCN_paras.items() if k != '__name'}
        if args.is_attr_available('Graph_paras'):
            graph_args = {k:v for k,v in args.Graph_paras.items() if k != '__name'}
        else:
            graph_args = None
        model = IntentGCN(input_shape, args.Model.out_channels, graph_args, **args.GCN_paras)
    
    elif args.Model.model == 'multi_laterfusion_pd':
        assert len(args.Model.sub_model_paths) == len(args.Data.features)
        assert isinstance(args.Model.sub_model_paths, list)

        submodels = []
        for i, path in enumerate(args.Model.sub_model_paths):
            assert os.path.exists(path)

            json_path = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.json')][0]

            submodel_args = JsonConfig(json_path)
            GCN_paras = {k:v for k,v in submodel_args.GCN_paras.items() if k != '__name'}
            if submodel_args.is_attr_available('Graph_paras'):
                graph_args = {k:v for k,v in submodel_args.Graph_paras.items() if k != '__name'}
            else:
                graph_args = None
            submodel = IntentGCN(input_shape[i], submodel_args.Model.out_channels, graph_args, **GCN_paras)
            
            submodel.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
            submodels.append(submodel)

        model = Multi_LaterFusion_PD(input_shape, args.Model.out_channels, submodels, args.Model.sub_model_detach, **args.Model_paras.to_dict())

    model = model.to(device)
    assert args.Model.load_model_path is not None, 'Please specify the model path to load'
    model.load_state_dict(torch.load(args.Model.load_model_path, map_location=device))
    print(f'[Info] Load model from {args.Model.load_model_path}')
    print(f'[Info] Model parameters: {get_param_num(model)}')

    model.eval()
    preds, labels = [], []

    for data in test_loader:
        inputs = [d.to(device) if type(d)!=list else [e.to(device) for e in d] for d in data[0]]

        if args.Optim.trim and np.random.randint(10) >= 5:
            if inputs[0].size(1) == 16:
                crop_size = np.random.randint(2, 11)
            elif inputs[0].size(1) == 32 or inputs[0].size(1) == 62:
                crop_size = np.random.randint(2, 21)
            else:
                raise(ValueError("Unknown input size: {}".format(inputs[0].size(1))))
            inputs = [d[:,-crop_size:] if d.size(1) == inputs[0].size(1) else d for d in inputs]
            
        y = data[1].to(device)
        
        outputs = model(*inputs)

        if type(outputs) in [list, tuple]:
                outputs = outputs[0]

        preds.extend(outputs.cpu().detach().numpy())
        # labels.extend(y[0].cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    preds = np.array(preds).argmax(axis=1)
    labels = np.array(labels)

    result = result_ana(labels, preds)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IntentGCN')

    parser.add_argument('--ckps_path', type=str, help='path to load checkpoints')
    args = parser.parse_args()

    config_path = os.path.join(args.ckps_path, 'config.json')
    config = JsonConfig(config_path)
    config.Model.load_model_path = os.path.join(args.ckps_path, 'model.pkl')

    main(config)