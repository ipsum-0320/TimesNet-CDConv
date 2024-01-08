import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")  # 忽略 RuntimeWarning。


def generate_fig(arg_root, arg_file, data_class):
    date_value_map = {}
    df = pd.read_excel(os.path.join(arg_root, arg_file))
    for j, excel_date in enumerate(df['date']):
        if excel_date not in date_value_map:
            date_value_map[excel_date] = 0
        date_value_map[excel_date] = date_value_map[excel_date] + (
                    df['total'][j] - df['available'][j] - df['unavailable'][j])

    x = np.array(list(reversed(list(date_value_map.keys()))))
    y = np.array(list(reversed(list(date_value_map.values()))))
    old_y = np.array(list(reversed(list(date_value_map.values()))))
    x = [item[11:-3] for _, item in enumerate(x)]
    for i in range(len(y)):
        if i == 0:
            if abs(y[i] - y[i + 1]) >= 100:
                # 异常数据，相邻时刻的实例数相差 100 及以上。
                y[i] = y[i + 1]
        elif i == len(y) - 1:
            if abs(y[i] - y[i - 1]) >= 100:
                y[i] = y[i - 1]
        else:
            boundary = (y[i - 1] + y[i + 1]) / 2
            if (y[i] > y[i - 1] and y[i] > y[i + 1] and abs(y[i] - boundary) >= 100) or (
                    y[i] < y[i - 1] and y[i] < y[i + 1] and abs(y[i] - boundary) >= 100):
                y[i] = boundary

    plt.figure(figsize=(18, 12))
    plt.plot(old_y, label="original-available", color="red")
    plt.plot(y, label="processed-available", color="blue")
    plt.xticks(range(0, len(x), 120), x[::120], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.savefig('./data_img/{}_img/{}.png'.format(data_class, arg_file[5:10]))
    plt.clf()


if __name__ == '__main__':
    original_data_class = 'train'
    for root, dirs, files in os.walk('./excel-original/{}/'.format(original_data_class)):
        cur = 0
        for file in files:
            generate_fig(root, file, original_data_class)
            cur = cur + 1
            print("generate data_img: {} / {}".format(cur, len(files)))

    print("generate data_img down.")
