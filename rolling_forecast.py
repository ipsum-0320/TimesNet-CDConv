import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import os
import pandas as pd



def generate_rolling_fig(dates, preds_val, trues_val, name, day_str, interval):
    plt.figure(figsize=(24, 16))
    plt.plot(dates, trues_val, label="true_value", color="red")
    plt.plot(dates, preds_val, label="predicted_value", color="blue")

    for i in range(len(preds_val)):
        if i % interval == 0:
            selected_date = dates[i]
            selected_pred_val = preds_val[i]
            plt.plot(selected_date, selected_pred_val, 'bo', markersize=10)
            plt.vlines(selected_date, ymin=0, ymax=selected_pred_val, colors='gray', linestyles='dashed')

    plt.xticks(range(0, len(dates), 60), dates[::60], fontsize=20, rotation=60)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    day_of_week = datetime.strptime(day_str, "%Y-%m-%d").strftime("%A")
    plt.title("{} - {} / {}%".format(day_str, day_of_week, round(100 - calculate_mape(trues_val, preds_val), 2)),
              fontsize=26,
              fontweight='bold')
    plt.savefig('./data_img/rolling_img/{}/{}.png'.format(name, day_str))
    plt.clf()


def calculate_mape(trues_val, preds_val):
    n = len(trues_val)
    mape = sum(abs((a - p) / a) for a, p in zip(trues_val, preds_val) if a != 0) / n
    return mape * 100


if __name__ == '__main__':
    TimesNet_all_30 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl180_ll180_pl30_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'
    TimesNet_all_60 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl180_ll180_pl60_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'

    TimesNet_all_30_pred = np.load('./results/{}/pred.npy'.format(TimesNet_all_30))
    TimesNet_all_60_pred = np.load('./results/{}/pred.npy'.format(TimesNet_all_60))

    TimesNet_all_30_true = np.load('./results/{}/true.npy'.format(TimesNet_all_30))
    TimesNet_all_60_true = np.load('./results/{}/true.npy'.format(TimesNet_all_60))

    date_info = [  # TODO 需要查看 test.csv 确定数据是否完整，即是否是从 00:00-23:59。
        [None, "00:00", "23:59", 0],
        [None, "00:00", "23:59", 1440],
        [None, "00:00", "23:59", 2880],
        [None, "00:00", "23:59", 4320],
        [None, "00:00", "23:59", 5760],
        [None, "00:00", "23:59", 7200],
        [None, "00:00", "23:59", 8640]
    ]

    for root, dirs, files in os.walk('./excel-original/test/'):
        index = 0
        for file in files:
            df = pd.read_excel(os.path.join(root, file))
            date_str = df['date'][0][0:10]
            date_info[index][0] = date_str
            index = index + 1

    for date_info_index in range(len(date_info)):
        pred_30_list = []
        pred_60_list = []
        true_list = []
        times = []

        start_time_str = date_info[date_info_index][1]
        end_time_str = date_info[date_info_index][2]

        start_time = datetime.strptime(start_time_str, "%H:%M")
        end_time = datetime.strptime(end_time_str, "%H:%M")

        current_time = start_time
        while current_time <= end_time:
            times.append(current_time.strftime("%H:%M"))
            current_time += timedelta(minutes=1)

        all_min = len(times)

        while len(pred_30_list) < all_min:
            cur_len = len(pred_30_list)
            for _, preds in enumerate(TimesNet_all_30_pred[cur_len + date_info[date_info_index][3]]):
                if len(pred_30_list) < all_min:
                    pred_30_list.append(preds[0])
                else:
                    break

        while len(pred_60_list) < all_min:
            cur_len = len(pred_60_list)
            for _, preds in enumerate(TimesNet_all_60_pred[cur_len + date_info[date_info_index][3]]):
                if len(pred_60_list) < all_min:
                    pred_60_list.append(preds[0])
                else:
                    break
            for _, trues in enumerate(TimesNet_all_60_true[cur_len + date_info[date_info_index][3]]):
                if len(true_list) < all_min:
                    true_list.append(trues[0])
                else:
                    break

        generate_rolling_fig(times, pred_30_list, true_list, "30", date_info[date_info_index][0], 30)
        generate_rolling_fig(times, pred_60_list, true_list, "60", date_info[date_info_index][0], 60)

        print("{} done.".format(date_info[date_info_index][0]))
