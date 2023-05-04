from data_loading.data_loader import *
from torch.utils.data import ConcatDataset

# 다음 함수 len(aug_list)=7일 때만 가능 (추후 일반화 시켜야 함..)
def augmentation(traindf, trainset, label, img_dir, size, tf, aug_list):
    all_datasets = [trainset]

    for i, aug in enumerate(aug_list):
        cls1_df = traindf[traindf[label] == 1]
        cls0_df = traindf[traindf[label] == 0]
        cls0_range = round(len(cls0_df) / len(aug_list))      # 클래스 불균형 때문에 cls0은 최종적으로 x2배만 되도록 aug

        j = i + 1
        if aug == 'hflip':
            cls1_df = cls1_df[:int(len(cls1_df) / 2)]
            cls0_df = cls0_df[cls0_range * i:(cls0_range * j) - int(cls0_range / 2)]  # 0:(96-47)
        elif aug == 'vflip':
            cls1_df = cls1_df[int(len(cls1_df) / 2):]
            cls0_df = cls0_df[(cls0_range * i) - int(cls0_range / 2):cls0_range * j]
        else:
            cls0_df = cls0_df[cls0_range * i:cls0_range * j]

        cls1_aug_img_name = [name[:-4] + '_{}.png'.format(aug) for name in list(cls1_df['img_name'])]
        cls1_df['img_name'] = cls1_aug_img_name
        cls0_aug_img_name = [name[:-4] + '_{}.png'.format(aug) for name in list(cls0_df['img_name'])]
        cls0_df['img_name'] = cls0_aug_img_name

        augsets_1 = CombineDataset(cls1_df, 'img_name', label, img_dir, input_size=size, transform=tf)
        augsets_0 = CombineDataset(cls0_df, 'img_name', label, img_dir, input_size=size, transform=tf)
        all_datasets.append(augsets_1)
        all_datasets.append(augsets_0)
    aug_train_dataset = ConcatDataset(all_datasets)
    #print('#### fold_train + aug_train datasets의 개수 :', len(all_datasets))  # 15개 나와야 함
    #print('#### aug한 fold_train_dataset의 개수 :', len(aug_train_dataset))
    return aug_train_dataset


def scaled_datasets(train_df, valid_df, test_df, scaler, continuous_feat):

    if continuous_feat == None:
        return train_df, valid_df, test_df

    else:
        scaled_train, scaled_valid, scaled_test = scaling(train_df, valid_df, test_df, scaler, continuous_feat)   # scaler = StandardScaler()

        scaled_train_df = pd.concat([train_df, scaled_train], axis=1)
        scaled_train_df = scaled_train_df.drop(continuous_feat, axis=1)

        scaled_valid_df = pd.concat([valid_df, scaled_valid], axis=1)
        scaled_valid_df = scaled_valid_df.drop(continuous_feat, axis=1)

        scaled_test_df = pd.concat([test_df, scaled_test], axis=1)
        scaled_test_df = scaled_test_df.drop(continuous_feat, axis=1)

        return scaled_train_df, scaled_valid_df, scaled_test_df


def scaling(train, valid, test_df, scaler, continuous_feat):
    contin_train = train[continuous_feat]
    contin_valid = valid[continuous_feat]
    contin_test = test_df[continuous_feat]

    fitted_scaler = scaler.fit(contin_train)

    scaled_t = fitted_scaler.fit_transform(contin_train)   # scaled_t : numpy.ndarray
    scaled_t = pd.DataFrame(scaled_t, columns=['scaled_'+feat for feat in continuous_feat])

    scaled_v = fitted_scaler.fit_transform(contin_valid)
    scaled_v = pd.DataFrame(scaled_v, columns=['scaled_'+feat for feat in continuous_feat])

    scaled_te = fitted_scaler.fit_transform(contin_test)
    scaled_te = pd.DataFrame(scaled_te, columns=['scaled_'+feat for feat in continuous_feat])

    scaled_train = get_scaled_df(scaled_t)
    scaled_valid = get_scaled_df(scaled_v)
    scaled_test = get_scaled_df(scaled_te)

    return scaled_train, scaled_valid, scaled_test


def get_scaled_df(scaled_df):
    for i in range(len(scaled_df.columns)):
        values = scaled_df[scaled_df.columns[i]]
        for j in range(len(values)):
            if str(values[j]) == 'nan':
                values[j] = 0.0
    return scaled_df