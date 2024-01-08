import numpy as np

if __name__ == '__main__':
    TimesNet_all_30 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl180_ll180_pl30_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'
    TimesNet_all_60 = 'long_term_forecast_TimesNet-CDConv_TimesNet_custom_ftMS_sl180_ll180_pl60_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0'

    TimesNet_all_30_metrics = np.load('./results/{}/metrics.npy'.format(TimesNet_all_30))
    TimesNet_all_60_metrics = np.load('./results/{}/metrics.npy'.format(TimesNet_all_60))

    TimesNet_all_30_times = np.load('./results/{}/times.npy'.format(TimesNet_all_30))
    TimesNet_all_60_times = np.load('./results/{}/times.npy'.format(TimesNet_all_60))

    metrics = [
        TimesNet_all_30_metrics[3],
        TimesNet_all_60_metrics[3],
    ]
    np.savetxt("./metrics_long_term_forecast.csv", metrics, delimiter=',', fmt='%.6f', header="TimesNet_mape")

    times = [
        TimesNet_all_30_times,
        TimesNet_all_60_times,
    ]
    np.savetxt("./times_long_term_forecast.csv", times, delimiter=',', fmt='%.3f', header="TimesNet_sum_time,"
                                                                                          "TimesNet_iterations,"
                                                                                          "TimesNet_avg_time")


