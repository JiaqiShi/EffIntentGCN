from utils.data_utils import representation_trans

from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

def get_JAAD_dataloader(data_path, features, batch_size=2, **kw):

    print(f'[Info] Loading JAAD dataset from {data_path}...')

    with open(data_path, 'rb') as f:
        datas = pickle.load(f)

    train_set, val_set, test_set = [JAAD_dataset(datas, features, sub_set, **kw) for sub_set in [0, 1, 2]]

    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

class JAAD_dataset(Dataset):

    def __init__(self, datas, features, sub_set=0, **kw):
        super().__init__()

        self.sub_set = sub_set
        
        assert self.sub_set in [0, 1, 2]
        data = datas[sub_set]

        self.labels = data[1]

        self.len = len(self.labels)

        self.feature_list = []

        if isinstance(features, list):
            self.feature_list = []
            for feature in features:
                self.feature_list.append(self._get_features(data, feature, **kw))
        elif isinstance(features, str):
            if ',' in features:
                self.feature_list = []
                for feature in features.split(','):
                    self.feature_list.append(self._get_features(data, feature, **kw))
            else:
                self.feature_list = [self._get_features(data, features, **kw)]
        else:
            raise ValueError(f'[ERROR] feature {features} not exists.')

    def _get_features(self, data, feature, **kw):
        if feature == 'pose2d':
            if len(data[0][0].shape) == 3 and data[0][0].shape[-1] == 36:
                return data[0][0].reshape(data[0][0].shape[0], data[0][0].shape[1], 18, 2)
            else:
                return data[0][0]
        elif feature in ['dirvec2d', 'dirvec_norm2d', 'temp_disp2d']:
            f_repre = []
            for d in self._get_features(data, 'pose2d').copy():
                f_repre.append(representation_trans(d, source_type='skeleton', target_type=feature[:-2], dataset='JAAD'))
            return np.array(f_repre)
        else:
            raise ValueError(f'[ERROR] feature {feature} not exists.')

    def __len__(self):
        return self.len
    
    def get_shape(self):    
        one_x = self[0][0]
        shapes = []
        for one_inp in one_x:
            if type(one_inp) == list:
                shapes.append(one_inp[0].shape)
            else:
                shapes.append(one_inp.shape)
        return shapes
        

    def __getitem__(self, index):
        d_one = [d[index] for d in self.feature_list if d is not None]
        return d_one, self.labels[index][0]
    
if __name__ == '__main__':
    with open('./data/jaad_beh.pkl', 'rb') as f:
        datas = pickle.load(f)
    dataset = JAAD_dataset(datas, sub_set=0)
    print(dataset.get_shape())