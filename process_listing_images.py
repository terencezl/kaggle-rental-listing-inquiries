import sys
import numpy as np
import pandas as pd
import cv2
from glob import glob

# parallel submission text generator
def get_parallel_jobs(scriptname, start, end, n_core=16):
    endpoints = np.linspace(start, end, n_core + 1).astype(int)
    for idx in range(len(endpoints) - 1):
        print('python process_images.py', str(endpoints[idx]), str(endpoints[idx + 1]), '&')
        print('sleep 3')
    print('wait')

# ==============================================================================
image_path_list = glob('images/*/*.jpg')

images = pd.Series(image_path_list)
images = images.rename('path').to_frame()
images[['listing_id', 'image_num']] = images.path.str.split('/', expand=True)[[1, 2]]
images.to_csv('images_basic_info.csv')

# ==============================================================================

images = pd.read_csv('images_basic_info.csv', index_col=0)

cols_to_add = ['width', 'height',
               'b_med', 'g_med', 'r_med',
               'b_mean', 'g_mean', 'r_mean',
               'b_std', 'g_std', 'r_std',
               'b_min', 'g_min', 'r_min',
               'b_max', 'g_max', 'r_max',
               'h_med', 's_med', 'v_med',
               'h_mean', 's_mean', 'v_mean',
               'h_std', 's_std', 'v_std',
               'h_min', 's_min', 'v_min',
               'h_max', 's_max', 'v_max',
               'g_med', 'g_mean', 'g_std', 'g_min', 'g_max'
               ]

for col in cols_to_add:
    images[col] = 0

for idx, row in images.iterrows():
    try:
        image = cv2.imread(row.path)
        prop_list = list(image.shape[:-1])

        prop_list.extend(np.median(np.median(image, axis=0), axis=0))
        prop_list.extend(image.mean(axis=0).mean(axis=0))
        prop_list.extend(image.std(axis=0).std(axis=0))
        prop_list.extend(image.min(axis=0).min(axis=0))
        prop_list.extend(image.max(axis=0).max(axis=0))

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        prop_list.extend(np.median(np.median(image_hsv, axis=0), axis=0))
        prop_list.extend(image_hsv.mean(axis=0).mean(axis=0))
        prop_list.extend(image_hsv.std(axis=0).std(axis=0))
        prop_list.extend(image_hsv.min(axis=0).min(axis=0))
        prop_list.extend(image_hsv.max(axis=0).max(axis=0))

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prop_list.append(np.median(np.median(image_gray, axis=0), axis=0))
        prop_list.append(image_gray.mean())
        prop_list.append(image_gray.std())
        prop_list.append(image_gray.min(axis=0).min(axis=0))
        prop_list.append(image_gray.max(axis=0).max(axis=0))

        images.loc[idx, cols_to_add] = prop_list
    except Exception as e:
        print(idx, row.path, e)

images.to_csv('images.csv')

# ==============================================================================
# combine
# images = pd.concat([pd.read_csv(i, index_col=0) for i in glob('images_dfs/images_*.csv')]).sort_index()
# images = images[images.image_num.str.contains('.jpg')].reset_index()
