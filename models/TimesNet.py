import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import TemporalConvNet
import pandas as pd
from prophet import Prophet
from models.TrendLTSM import LSTM

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.linear_dim = (self.seq_len + self.pred_len) * configs.d_model
        self.TemporalConvNet_intraperiod = TemporalConvNet(configs.d_ff,
                                                           [configs.d_model, configs.d_model, configs.d_model])
        self.TemporalConvNet_interperiod = TemporalConvNet(configs.d_ff,
                                                           [configs.d_model, configs.d_model, configs.d_model])
        self.adaptive_avg_pool_2d_intraperiod = nn.AdaptiveAvgPool2d((self.seq_len + self.pred_len, 1))
        self.adaptive_avg_pool_2d_interperiod = nn.AdaptiveAvgPool2d((self.seq_len + self.pred_len, 1))
        self.intraperiod_net = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.interperiod_net = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out_dim = out.size()
            intraperiod_res = []
            interperiod_res = []
            for j in range(out_dim[3]):
                intraperiod_res.append(self.TemporalConvNet_intraperiod(out[:, :, :, j]))
            for k in range(out_dim[2]):
                interperiod_res.append(self.TemporalConvNet_interperiod(out[:, :, k, :]))
            intraperiod_out = torch.stack(intraperiod_res, dim=3)
            interperiod_out = torch.stack(interperiod_res, dim=2)
            intraperiod_out = self.adaptive_avg_pool_2d_intraperiod(intraperiod_out)
            interperiod_out = self.adaptive_avg_pool_2d_interperiod(interperiod_out)
            intraperiod_out = intraperiod_out.permute(0, 2, 3, 1).reshape(B, -1)
            interperiod_out = interperiod_out.permute(0, 2, 3, 1).reshape(B, -1)
            out = self.intraperiod_net(intraperiod_out) + self.interperiod_net(interperiod_out)
            out = out.reshape(B, -1, N)
            # reshape back
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.lstm = LSTM(input_size=configs.enc_in, hidden_size=configs.enc_in, num_layers=2,
                         output_size=configs.enc_in, batch_size=configs.batch_size / 2, output_step=configs.pred_len)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, ___x_dec, ___x_mark_dec, date_str):
        # x_enc 是过去 360 分钟的数据，x_mark_enc 是时间特征。
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 数据合并
            df_value_list = [
                pd.DataFrame(x_enc[i].cpu().numpy(),
                             columns=['isMonday', 'isTuesday', 'isWednesday', 'isThursday', 'isFriday',
                                      'isSaturday', 'isSunday', 'hour', 'isRest', 'target'])
                for i in range(x_enc.shape[0])]
            df_date_list = [pd.DataFrame(date_str[i], columns=['date']) for i in range(date_str.shape[0])]

            df_merged_list = [
                pd.concat([df_date_list[i].reset_index(drop=True), df_value_list[i].reset_index(drop=True)], axis=1)
                for i in range(len(df_value_list))
            ]

            device = x_mark_enc.device
            for i in range(x_enc.shape[0]):
                x_enc[i].to(device)

            # 时序分解
            holidays = pd.DataFrame({
                'holiday': ['national_day'],
                'ds': pd.to_datetime(
                    ['2023-10-04']),
                'lower_window': [0],  # 在节假日当天开始影响
                'upper_window': [4]  # 在节假日后的第4天结束影响
            })

            x_enc_trend = x_enc.clone()

            # Prophet 无法直接处理批次数据，需要使用 for 循环来处理。
            for i in range(len(df_merged_list)):
                df = df_merged_list[i]
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    holidays=holidays)  # 启用年、周、日周期性
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')  # 手动指定格式，以加快解析速度
                df.rename(columns={'date': 'ds', 'target': 'y'}, inplace=True)  # 确保日期列名为 'ds'，目标列名为 'y'
                model.fit(df)

                # 分解数据
                future = model.make_future_dataframe(periods=0)  # 不扩展未来，仅分解当前数据
                forecast = model.predict(future)

                # 提取趋势和季节性序列
                trend_series = forecast['trend']
                weekly_series = forecast['weekly']  # 每周季节性
                yearly_series = forecast['yearly']  # 每年季节性
                daily_series = forecast['daily']  # 每日季节性
                combined_seasonal_series = weekly_series + yearly_series + daily_series

                combined_seasonal_series_tensor = torch.tensor(combined_seasonal_series, dtype=x_enc[i].dtype)
                x_enc[i][:, -1] = combined_seasonal_series_tensor[:]
                trend_series_tensor = torch.tensor(trend_series, dtype=x_enc_trend[i].dtype)
                x_enc_trend[i][:, -1] = trend_series_tensor[:]

            # 趋势项预测
            trend_dec_out = self.lstm(x_enc_trend)

            # 季节项预测
            dec_out = self.forecast(x_enc, x_mark_enc)
            # dec_out 是未来 60 分钟的预测值。
            dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return trend_dec_out + dec_out
        return None
