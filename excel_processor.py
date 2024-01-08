import math
import os
import datetime

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import copy
from scipy import stats
from chinese_calendar import is_holiday


def extract_date_from_filename(filename):
    start_index = filename.find('(') + 1
    end_index = filename.find(')')
    date_str = filename[start_index:end_index]
    date = datetime.datetime.strptime(date_str[0:5], '%m-%d')
    return date


def excel_processor():
    np.seterr(invalid='ignore')
    slope_flag = 0  # [0, 1, 2]，0 表示前后 10 个点，1 表示前 10 个点，2 表示后 10 个点。
    pd.set_option('mode.chained_assignment', None)
    original_data_class = ['train', 'test']

    for _, data_class in enumerate(original_data_class):
        date_list = []
        value_list = []
        print("{} excel inputting...".format(data_class))

        cur_pointer = 1
        for root, dirs, files in os.walk('./excel-original/{}/'.format(data_class)):
            last_pointer = len(files)
            sorted_files = sorted(files, key=extract_date_from_filename)
            for file in sorted_files:
                df = pd.read_excel(os.path.join(root, file))
                print("[{}/{}] input {}...".format(cur_pointer, last_pointer, file))
                date_value_map = {}
                for j, excel_date in enumerate(df['date']):
                    if excel_date not in date_value_map:
                        date_value_map[excel_date] = 0
                    date_value_map[excel_date] = date_value_map[excel_date] + (df['total'][j] - df['available'][j] - df['unavailable'][j])

                tmp_date_list = []
                tmp_value_list = []
                for key, value in date_value_map.items():
                    tmp_date_list.append(key)
                    tmp_value_list.append(value)

                if sorted(tmp_date_list) != tmp_date_list:
                    tmp_date_list.reverse()
                    tmp_value_list.reverse()

                for _, date in enumerate(tmp_date_list):
                    date_list.append(date)
                for _, value in enumerate(tmp_value_list):
                    value_list.append(value)
                cur_pointer = cur_pointer + 1

        for i in range(len(value_list)):
            if i == 0:
                if abs(value_list[i] - value_list[i + 1]) >= 100:
                    # 异常数据，相邻时刻的实例数相差 100 及以上。
                    value_list[i] = value_list[i + 1]
            elif i == len(value_list) - 1:
                if abs(value_list[i] - value_list[i - 1]) >= 100:
                    value_list[i] = value_list[i - 1]
            else:
                boundary = (value_list[i - 1] + value_list[i + 1]) / 2
                if (value_list[i] > value_list[i - 1] and value_list[i] > value_list[i + 1] and abs(
                        value_list[i] - boundary) >= 100) or (
                        value_list[i] < value_list[i - 1] and value_list[i] < value_list[i + 1] and abs(
                        value_list[i] - boundary) >= 100):
                    value_list[i] = boundary

        df_map = {
            "date": date_list,
            "value": value_list
        }
        df = DataFrame(df_map)
        print("input done. featuring...")

        df['date'] = pd.to_datetime(df['date'])
        df_with_day_of_week = copy.deepcopy(df)
        df_with_day_of_week['day_of_week'] = df_with_day_of_week['date'].dt.day_name()

        df['isMonday'] = "0"
        df['isTuesday'] = "0"
        df['isWednesday'] = "0"
        df['isThursday'] = "0"
        df['isFriday'] = "0"
        df['isSaturday'] = "0"
        df['isSunday'] = "0"
        df['hour'] = "00"  # 用于记录小时特征，
        df['slope'] = "0"  # 用于记录斜率特征。
        df['range'] = "0"  # 用于记录极差。
        df['deviation'] = "0"  # 用于记录标准差。
        df['isRest'] = "0"

        df['target'] = df['value']
        df.drop("value", axis=1, inplace=True)

        for i, item in enumerate(df_with_day_of_week['day_of_week']):
            df["is{}".format(item)][i] = "1"

        for i, item in enumerate(df['date']):
            df['hour'][i] = item.hour
            datetime_item = datetime.date(item.year, item.month, item.day)
            if is_holiday(datetime_item):
                df['isRest'][i] = "1"

        boundary = len(df['target'])
        for i, item in enumerate(df['target']):
            if slope_flag == 0:
                x = list(range((i - 10 if i - 10 >= 0 else 0), (i + 11 if i + 11 <= boundary else boundary)))
                y = df['target'][(i - 10 if i - 10 >= 0 else 0):(i + 11 if i + 11 <= boundary else boundary)].values
            elif slope_flag == 1:
                x = list(range((i - 10 if i - 10 >= 0 else 0), (i + 1 if i + 1 <= boundary else boundary)))
                y = df['target'][(i - 10 if i - 10 >= 0 else 0):(i + 1 if i + 1 <= boundary else boundary)].values
            else:
                x = list(range(i, (i + 11 if i + 11 <= boundary else boundary)))
                y = df['target'][i:(i + 11 if i + 11 <= boundary else boundary)].values
            slope = stats.linregress(x, y).slope
            _range = np.max(y) - np.min(y)
            deviation = np.std(y)
            if math.isnan(slope):
                slope = 0
            df['slope'][i] = "{}".format(round(slope, 2))
            df['range'][i] = "{}".format(round(_range, 2))
            df['deviation'][i] = "{}".format(round(deviation, 2))

        print("feature done. outputting...")
        df.to_csv("./dataset/TimesNet-CDConv/{}.csv".format(data_class), index=False)
        print("output {}.csv done.".format(data_class))

    print("all outputs done.")
