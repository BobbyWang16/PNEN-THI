# 计算影像组学特征
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor 

# 特征计算函数
def catch_features(imagePath, maskPath, label_num=2):
    settings = {}
    settings['binWidth'] = 25  
    settings['label'] = label_num
    settings['Interpolator'] = 3
    settings['resampledPixelSpacing'] = [1, 1, 1]
    settings['normalize'] = True
    settings['normalizeScale'] = 255

    # 提取特征
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    result = extractor.execute(imagePath, maskPath)
    return result

# 计算整个参数
path = "./data/"
data_path = path + 'crop_V'
mask_path = path + 'crop_ROI'

namelist = os.listdir(mask_path)

# 根据列名来找影像及勾画文件路径
rad_df = pd.DataFrame()
for name in namelist:
    print(name)
    imagePath = os.path.join(data_path, name)
    maskPath = os.path.join(mask_path, name)
    result_mask = catch_features(imagePath, maskPath, label_num=2)
    rad_series = pd.DataFrame(result_mask.values(), index=result_mask.keys(), columns=[name.replace('.nii.gz', '')])
    # 把rad_series转置，然后拼接
    rad_df = pd.concat([rad_df, rad_series], axis=1)


rad_feature = rad_df.transpose()
rad_feature.to_csv(path + 'V_radiomics.csv')
