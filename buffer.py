import numpy as np
from matplotlib import pyplot as plt


def buffer(history_pred, history_true, cur_pred):
    # 基于过往 120 分钟的数据做 Buffer。
    # weight 是权重，0 取 mean，1 取 max，weight 在 [0, 正无穷] 之间。
    history = []
    for i in range(len(history_pred)):
        if history_true[i] > history_pred[i]:
            history.append((history_true[i] - history_pred[i]) / history_pred[i])

    mean_history = np.mean(history)
    max_history = np.max(history)
    cur_pred_buffered = []
    for i, pred in enumerate(cur_pred):
        cur_pred_buffered.append(pred * (1 + mean_history))
    return cur_pred_buffered


def generate_buffer_fig(pred_val, true_val, buffered_pred_val, interval, cur):
    plt.figure(figsize=(24, 16))
    interval_x = range(interval)
    plt.plot(interval_x, true_val, label="true_value", color="red")
    plt.plot(interval_x, pred_val, label="predicted_value", color="blue")
    plt.plot(interval_x, buffered_pred_val, label="buffered_predicted_value", color="green")

    for i in range(len(pred_val)):
        selected_pred_val = max(pred_val[i], true_val[i], buffered_pred_val[i])
        plt.vlines(interval_x, ymin=0, ymax=selected_pred_val, colors='gray', linestyles='dashed')

    plt.xticks(range(0, len(interval_x)), fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)

    original_cover_num = 0
    buffered_cover_num = 0
    for i in range(len(pred_val)):
        if pred_val[i] >= true_val[i]:
            original_cover_num = original_cover_num + 1
        if buffered_pred_val[i] >= true_val[i]:
            buffered_cover_num = buffered_cover_num + 1

    plt.title("{:.2f}% / {:.2f}%".format(original_cover_num * 100 / len(pred_val), buffered_cover_num * 100 / len(pred_val)), fontsize=26, fontweight='bold')
    plt.savefig('./data_img/buffered_img/{}_{}.png'.format(cur, interval))
    plt.clf()


if __name__ == '__main__':
    cur = 12
    interval = 30

    TimesNet_all_30 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl120_ll120_pl30_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'
    TimesNet_all_60 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl120_ll120_pl60_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'

    TimesNet_all_30_pred = np.load('./results/{}/pred.npy'.format(TimesNet_all_30))
    TimesNet_all_60_pred = np.load('./results/{}/pred.npy'.format(TimesNet_all_60))

    TimesNet_all_30_true = np.load('./results/{}/true.npy'.format(TimesNet_all_30))
    TimesNet_all_60_true = np.load('./results/{}/true.npy'.format(TimesNet_all_60))

    cur_pred = []
    cur_true = []
    history_pred = []
    history_true = []
    if interval == 30:
        for i, pred in enumerate(TimesNet_all_30_pred[cur * interval]):
            cur_pred.append(pred[0])
        for i, pred in enumerate(TimesNet_all_30_pred[(cur - 2) * interval]):
            history_pred.append(pred[0])
        for i, pred in enumerate(TimesNet_all_30_pred[(cur - 1) * interval]):
            history_pred.append(pred[0])
        for i, true in enumerate(TimesNet_all_30_true[cur * interval]):
            cur_true.append(true[0])
        for i, true in enumerate(TimesNet_all_30_true[(cur - 2) * interval]):
            history_true.append(true[0])
        for i, true in enumerate(TimesNet_all_30_true[(cur - 1) * interval]):
            history_true.append(true[0])
    elif interval == 60:
        for i, pred in enumerate(TimesNet_all_60_pred[cur * interval]):
            cur_pred.append(pred[0])
        for i, pred in enumerate(TimesNet_all_60_pred[(cur - 2) * interval]):
            history_pred.append(pred[0])
        for i, pred in enumerate(TimesNet_all_60_pred[(cur - 1) * interval]):
            history_pred.append(pred[0])
        for i, true in enumerate(TimesNet_all_60_true[cur * interval]):
            cur_true.append(true[0])
        for i, true in enumerate(TimesNet_all_60_true[(cur - 2) * interval]):
            history_true.append(true[0])
        for i, true in enumerate(TimesNet_all_60_true[(cur - 1) * interval]):
            history_true.append(true[0])

    cur_pred_buffered = buffer(history_pred, history_true, cur_pred)
    generate_buffer_fig(cur_pred, cur_true, cur_pred_buffered, interval, cur)

    print("buffer down.")
