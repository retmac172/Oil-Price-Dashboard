# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:52:05 2023

@author: burcu
"""
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output 

import re
import time
import os
import numpy as np
import csv
import json
import datetime
import csv
import pandas as pd
import math
import re
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import keras
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import dash_bootstrap_components as dbc
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from ast import literal_eval
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
from dash import dash_table

def calculate_ema(df, window):
    df['ema'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def calculate_macd(df, window_short, window_long, signal): ### short 12 long 26
    ema_short = df['Close'].ewm(span=window_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=window_long, adjust=False).mean()
    df['macd_line'] = ema_short - ema_long
    df['signal_line'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
    df['histogram'] = df['macd_line'] - df['signal_line']
    return df

def calculate_rsi(df, window):
    price_diff = df['Close'].diff(1)
    gain = price_diff.mask(price_diff < 0, 0)
    loss = -price_diff.mask(price_diff > 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    relative_strength = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + relative_strength))
    return df
def calculate_ma(df, window):
    df['MA'] = df['Close'].rolling(window).mean()
    return df

def calculate_momentum(df, window):
    df['Momentum'] = df['Close'].diff(window)
    return df

def calculate_williams_r(df, window):
    highest_high = df['High'].rolling(window).max()
    lowest_low = df['Low'].rolling(window).min()
    df['williams_r'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
    return df

def calculate_stochastic_oscillator(df, window, smooth_window):
    highest_high = df['High'].rolling(window).max()
    lowest_low = df['Low'].rolling(window).min()
    df['stochastic_k'] = (df['Close'] - lowest_low) / (highest_high - lowest_low) * 100
    df['stochastic_d'] = df['stochastic_k'].rolling(smooth_window).mean()
    return df
def calculate_obv(df,window):
    df['OBV'] = 0
    df.loc[df['Close'] > df['Close'].shift(1), 'OBV'] = df['Vol.']
    df.loc[df['Close'] < df['Close'].shift(1), 'OBV'] = -df['Vol.']
    df['OBV'] = df['OBV'].rolling(window=window).mean()
    return df
def calculate_bollinger_bands(df, window=20, std_dev=2):
    # Hareketli ortalama hesapla
    df['MA_BB'] = df['Close'].rolling(window=window).mean()
    
    # Hareketli standart sapma hesapla
    df['STD'] = df['Close'].rolling(window=window).std()
    
    # Üst bant hesapla
    df['Upper Band'] = df['MA_BB'] + std_dev * df['STD']
    
    # Alt bant hesapla
    df['Lower Band'] = df['MA_BB'] - std_dev * df['STD']
    df.drop(['STD'], axis=1, inplace=True)
    return df
def calculate_fibonacci_retracement(df, swing_high, swing_low):
    # Dönüş seviyelerini hesapla
    
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # Swing yüksek ve düşük fiyatları kullanarak Fibonacci Retracement seviyelerini hesapla
    diff = swing_high - swing_low
    retracements = [swing_high - level * diff for level in levels]
    
    # DataFrame'e Fibonacci Retracement seviyelerini ekle
    for i, level in enumerate(levels):
        df[f'Retracement {i}'] = retracements[i]
    
    return df
def calculate_average_true_range(df, period):
    # True Range sütunu oluşturma
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['True Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # ATR sütunu oluşturma
    df['ATR'] = df['True Range'].rolling(window=period).mean()
    
    # Kullanılan geçici sütunları kaldırma
    df.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'True Range'], axis=1, inplace=True)
    
    return df

def calculate_typical_price(df):
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    return df

def calculate_simple_moving_average(df, column, window):
    df['SMA'] = df[column].rolling(window=window).mean()
    return df

def calculate_mean_deviation(df, column, window):
    df['Mean Deviation'] = df[column].rolling(window=window).apply(lambda x: abs(x - x.mean()).mean())
    return df

def calculate_commodity_channel_index(df, period):
    # Tipik Fiyat hesaplama
    df = calculate_typical_price(df)
    
    # 20 günlük basit hareketli ortalama hesaplama
    df = calculate_simple_moving_average(df, 'Typical Price', period)
    df = calculate_mean_deviation(df, 'Typical Price', period)
    # CCI hesaplama
    df['CCI'] = (df['Typical Price'] - df['SMA']) / (0.015 * df['Mean Deviation'])
    
    # Kullanılan geçici sütunları kaldırma
    df.drop(['Typical Price', 'SMA', 'Mean Deviation'], axis=1, inplace=True)
    
    return df

def calculate_rate_of_change(df, period):

    
    df['ROC'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
    return df
def convert_volume(volume):
    if volume.endswith('K'):
        return float(volume[:-1]) * 1000
    elif volume.endswith('M'):
        return float(volume[:-1]) * 1000000
    else:
        return float(volume)
#############
df1=pd.read_csv('C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/future_2006.csv')
df2=pd.read_csv('C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/future2_2006.csv')
df_candle=pd.concat([df2, df1], ignore_index=True)
df_candle['Vol.']=df_candle['Vol.'].astype(str)
df_candle['Vol.']=df_candle['Vol.'].apply(convert_volume)
df_candle['Date'] = pd.to_datetime(df_candle['Date'])
df_candle['Vol.'].ffill(inplace=True)
df_candle.rename(columns={'Price': 'Close'}, inplace=True)
df_candle.drop(columns=['Change %'],inplace=True)
df_candle.sort_values(by='Date', inplace=True)
df_candle.reset_index(drop=True,inplace=True)
dfw=pd.read_csv('C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/future_weekly.csv')
dfw['Vol.']=dfw['Vol.'].astype(str)
dfw['Vol.']=dfw['Vol.'].apply(convert_volume)
dfw['Date'] = pd.to_datetime(dfw['Date'])
dfw['Vol.'].ffill(inplace=True)
dfw.dropna(inplace=True)
dfw.rename(columns={'Price': 'Close'}, inplace=True)
dfw.drop(columns=['Change %'],inplace=True)
dfw.sort_values(by='Date', inplace=True)
dfw.reset_index(drop=True,inplace=True)

df_4h=pd.read_excel("C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/generated_4h.xlsx")
df_1h=pd.read_excel("C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/generated_1h.xlsx")

fin_cols=['BTC','MSCI','DJIA','SP500 Price','Volin','XOM']
tech_ind=['EMA_5', 'EMA_10', 'EMA_20',
'MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'RSI_5', 'RSI_10',
'RSI_20', 'MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10',
'Momentum_20', 'Williams %R_5', 'Williams %R_10', 'Williams %R_20',
'Stochastic K', 'Stochastic D', 'OBV', 'OBV Average_5',
'OBV Average_10', 'OBV Average_20','Volume',]
comm_cols=['BDT','Gold Future','GSCI','Hot Oil Spot Price','Natural Gas Price','NY Harbor Gasoline Spot Price','Silver Price',
           'US Crude Stock','US Gulf','WTI Future']
exc_cols=['USD-CNY','USD-EUR','USD-TWD','US Dollar Index']
eco_ind=['FFR','US3Month','US 10Year Bond','USEPU']
default=['Price']

df=pd.read_excel('C:/Users/tayla/Desktop/burcu_tez/ts/DailyForecast/modeldata_2306.xlsx')
df.drop(columns=['Unnamed: 0','Open','High','Low'],inplace=True)
df['LD Price'] = df['Price']
df.set_index('Date', inplace=True)


all_features=['Financial Markets','Technical Indicators','Commodity Markets','Exchange Rates','Economic Indicators']
app = Dash('burcutez',external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title='WTI Futures Trading Decision Support System'
all_features_names=['Bitcoin Price','MSCI World','Dow Jones Industrial Index','S&P500','CBOE Volatility Index (VIX)','Exxon Mobil Corporation (XOM) Stock Price',
                    'Baltic Exchange Dirty Tanker Index','Gold Price','S&P GSCI Non Energy Index','Heating Oil Spot Price','Natural Gas Price','NY Harbor Gasoline Spot Price','Silver Price',
                    'US Crude Oil Stock (including SPR)','US Gulf Coast Gasoline Price','USD-CNY','USD-EUR','USD-TWD','US Dollar Index','US Federal Fund Rate,3-month U.S. Treasury Bill Rate',
                    'US 10-Year Bond Yield','US Economic Uncertainty Index (EPU)']
feature_codes=['BTC','MSCI','DJIA','SP500 Price','Volin','XOM','BDT','Gold Future','GSCI','Hot Oil Spot Price','Natural Gas Price','NY Harbor Gasoline Spot Price','Silver Price','US Crude Stock',
               'US Gulf','USD-CNY','USD-EUR','USD-TWD','US Dollar Index','US3Month','US 10Year Bond','USEPU']
# model_price=load_model('C:/Users/tayla/Desktop/burcu_tez/ts/dash/deneme.h5')
# model_trend=load_model('C:/Users/tayla/Desktop/burcu_tez/ts/dash/deneme_cat.h5')
#######################################################################################
def predictors():
    layout = html.Div([
        html.Div([ 
            html.H2("Model Predictors ", style={'text-align': 'center'}),
            ],),
        
        html.Div([
        dcc.RadioItems(id="Radio2",
                       options=[
                           {'label': 'Financal Markets', 'value': 1},
                           {'label': 'Technical Indicators', 'value': 2},
                           {'label': 'Commodity Market', 'value': 3},
                           {'label': 'Exchange Rates', 'value': 4},
                           {'label': 'Economic Indicators', 'value': 5},
                       ],
                       value=0,
                       style={'display': 'inline-block', 'width': '49%'}
                       ),
        html.Div(id='radio2-exp', children=[], style={"color": "black", "font-weight": "bold",'display': 'inline-block', 'width': '49%'}),
        ],),
        dcc.Dropdown(id="Features Graph",
                     options=[{'label': col, 'value': col} for col in all_features_names],
                     multi=True,
                     value=all_features_names[1],
                     style={'width': "60%"}
                     ),
        html.Div( 
            dcc.Graph(id='Predictor Graph', figure={}),
            ),
        
        ])
    @app.callback(
            [Output(component_id='radio2-exp', component_property='children'),
        
              Output(component_id='Predictor Graph', component_property='figure'),
              
                      ],
            [Input(component_id='Radio2', component_property='value'),
              Input(component_id='Features Graph', component_property='value'),
              ]
    )
    def update_predictors(radio2,graph_list):
        text=''
        if radio2 ==1:
            text="Bitcoin Price, MSCI World, Dow Jones Industrial Index, S&P500, CBOE Volatility Index (VIX), Exxon Mobil Corporation (XOM) Stock Price"
        elif radio2==2:
            text="EMA, MACD, RSI, MA, Momentum, Williams %R, Stochastic Oscillator, On-Balance Volume, Volume"
        elif radio2==3:
            text="Baltic Exchange Dirty Tanker Index,Gold Price, S&P GSCI Non Energy Index, Heating Oil Spot Price, Natural Gas Price, NY Harbor Gasoline Spot Price, Silver Price, US Crude Oil Stock (including SPR), US Gulf Coast Gasoline Price"
        elif radio2==4:
            text="USD-CNY, USD-EUR, USD-TWD, US Dollar Index"
        elif radio2==5:
            text="US Federal Fund Rate, 3-month U.S. Treasury Bill Rate, US 10-Year Bond Yield, US Economic Uncertainty Index (EPU)"
        fig = go.Figure()
        for i in range(len(all_features_names)):
            if all_features_names[i] in graph_list:
                fig.add_trace(go.Scatter(x=df.index, y=df[feature_codes[i]], mode='lines', name=all_features_names[i]))

        return (text,fig)
    return layout
def prediction():
        
    from dash import dcc, html, Input, Output, callback
    import plotly.express as px
    import pandas as pd
    from dash import dash_table
    #################### Burdan tablodaki yazılanları düzelt !!!

    f_data = [
    {'Feature Set Name': 'Financal Markets', 'Features': "Bitcoin Price,MSCI World,Dow Jones Industrial Index,S&P500,CBOE Volatility Index (VIX),Exxon Mobil Corporation (XOM) Stock Price"}, 
    {'Feature Set Name': 'Technical Indicators', 'Features': "EMA,MACD, RSI, MA, Momentum, Williams %R,Stochastic Oscillator, On-Balance Volume,Volume"},
    {'Feature Set Name': 'Commodity Market', 'Features': "Baltic Exchange Dirty Tanker Index,Gold Price,S&P GSCI Non Energy Index,Heating Oil Spot Price,Natural Gas Price,NY Harbor Gasoline Spot Price,Silver Price,US Crude Oil Stock (including SPR),US Gulf Coast Gasoline Price"},
    {'Feature Set Name': 'Exchange Rates', 'Features': "USD-CNY,USD-EUR,USD-TWD,US Dollar Index"} ,
    {'Feature Set Name': 'Economic Indicators', 'Features': "US Federal Fund Rate,3-month U.S. Treasury Bill Rate,US 10-Year Bond Yield,US Economic Uncertainty Index (EPU)"}]
    layout = html.Div([
    html.Div([
        html.H1("Price Forecasts of WTI Front Month Futures and WTI Spot ", style={'text-align': 'center'}),
        dcc.RadioItems(id="Radio",
                       options=[
                           {'label': 'Next Day Prediction', 'value': 1},
                           {'label': 'Next Week Prediction', 'value': 7},
                       ],
                       value=1
                       ),
        dcc.Dropdown(id="Features",
                     options=[{'label': col, 'value': col} for col in all_features],
                     multi=True,
                     value=all_features,
                     style={'width': "60%"}
                     ),
        
    ], ),
    html.Div([
    html.Div(id='Price-cont', children=[], style={"color": "red", "font-weight": "bold",'display': 'inline-block', 'width': '49%'}),
    html.Div(id='Price-cont-spot', children=[], style={"color": "red", "font-weight": "bold",'display': 'inline-block', 'width': '49%'}),
    ],),
 
    html.Div([
        dcc.Graph(id='Price-Pred', figure={}, style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='Price-Pred-spot', figure={}, style={'display': 'inline-block', 'width': '49%'}),
    ],),
    
    
])
    
    @app.callback(
        [Output(component_id='Price-cont', component_property='children'),
         Output(component_id='Price-cont-spot', component_property='children'),
          Output(component_id='Price-Pred', component_property='figure'),
          Output(component_id='Price-Pred-spot', component_property='figure'),
          
                  ],
        [Input(component_id='Radio', component_property='value'),
          Input(component_id='Features', component_property='value'),
          ]
    )
    
    def upgrade_app(radio,features):
        
        model_name='C:/Users/tayla/Desktop/burcu_tez/ts/models/'
        # c_model_name='C:/Users/tayla/Desktop/burcu_tez/ts/models/cat_'
        final_col=default
        add=0
        if 'Financial Markets' in features:
            final_col=final_col+fin_cols
            model_name=model_name+'_f'
            # c_model_name=c_model_name +'_f'
            add=add+0.290
        if 'Technical Indicators' in features:
            final_col=final_col+tech_ind
            model_name=model_name+'_t'
            # c_model_name=c_model_name +'_t'
            add=add+0.155
        if 'Commodity Markets' in features:
            final_col=final_col+comm_cols
            model_name=model_name+'_c'
            # c_model_name=c_model_name +'_c'
            add=add-0.385
        if 'Exchange Rates' in features:
            final_col=final_col+exc_cols
            model_name=model_name+'_ex'
            # c_model_name=c_model_name +'_ex'
            add=add+0.35
        if 'Economic Indicators' in features:
            final_col=final_col+eco_ind 
            model_name=model_name+'_ec'
            # c_model_name=c_model_name +'_ec'
            add=add-0.255
        
        if radio==1:
            
            # text_trend='Trend prediction for 1 day'
            model_name=model_name+'_1'
            # c_model_name=c_model_name +'_1'
            shf=-1
            timestamp=1

        elif radio==7:

            # text_trend='Trend prediction for 7 days'
            model_name=model_name+'_7'
            # c_model_name=c_model_name +'_7'
            shf=-5
            timestamp=7
        
            
        model_name=model_name+'.h5'
        dff=df.copy()
        dff=dff[final_col]
        model=load_model(model_name)
        X=dff.iloc[-1:]
        #X.set_index('Date', inplace=True)
        y_last=dff['Price'].iloc[-2]
        y = X['Price'].values.reshape(-1, 1)
        X.drop(columns='Price',inplace=True)
        scaler = MinMaxScaler()
        # X_cat=dff.iloc[-1:]
        # X_cat_scaled = scaler.fit_transform(X_cat)
        time=dff.index[-1]
        final_timestamp = time + pd.Timedelta(days=timestamp)
        final_timestamp=final_timestamp.strftime('%Y-%m-%d')
        text_price='WTI Fronth Month Future Forecast for  ' +str(final_timestamp)
        text_price2='WTI Spot Forecast for  ' +str(final_timestamp)
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y)
        y_pred = model.predict(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1))
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_value=y_pred_inv[0][0] 
        spot=np.round(y_value+add,3)
            
        
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = np.round(y_value, 2),
        delta = {'reference': y_last, 'relative': True},
        domain = {'x': [0, 0.5], 'y': [0.2, 1]}))
        
        
        fig2 = go.Figure()
        fig2.add_trace(go.Indicator(
        mode = "number+delta",
        value =  np.round(spot, 2),
        delta = {'reference': y_last+add/2, 'relative': True},
        domain = {'x': [0, 0.5], 'y': [0.2, 1]}))
        
        

          
        
        
        
        return(text_price,text_price2,fig,fig2)
    return layout
############################################################################
def sentiment():
    from dash import dash_table
    import plotly.express as px
    sent_data=pd.read_excel('C:/Users/tayla/Desktop/burcu_tez/ts/dash/0806_output.xlsx')

    layout = html.Div([

    dcc.DatePickerSingle(
        id='Date-picker',
        min_date_allowed=date(2011, 6, 30),
        max_date_allowed=date(2023, 3, 27),
        initial_visible_month=date(2023, 3, 7),
        date=date(2023, 3, 7),
    ),

    dash_table.DataTable(
        id='sentiment-table',
        columns=[{'name': 'News', 'id': 'News'}, {'name': 'Scores', 'id': 'Scores'}],
        data=[],
        style_data_conditional=[
                {
                    'if': {'column_id': 'News'},
                    'textAlign': 'left'
                },
                {
                    'if': {
                        'column_id': 'Scores',
                        'filter_query': '{Scores} > 0.25'
                    },
                    'backgroundColor': 'green',
                    'color': 'white'
                },
                {
                    'if': {
                        'column_id': 'Scores',
                        'filter_query': '{Scores} < -0.25'
                    },
                    'backgroundColor': 'red',
                    'color': 'white'
                },
                {
                    'if': {
                        'column_id': 'Scores',
                        'filter_query': '{Scores} >= -0.25 && {Scores} <= 0.25'
                    },
                    'backgroundColor': 'grey',
                    'color': 'white'
                }
            ],
            style_cell={
                'textAlign': 'left'
            }
      
        ),
    html.Div([
    
        html.H1("Cumulative Daily Score ", style={'text-align': 'center'})]),
    dcc.Graph(id='Sent_Score', figure={}),
        html.Div([
    html.Div([
        dcc.Graph(id='Sent_Hist', figure={}),
    ], style={'display': 'inline-block', 'width': '49%'}),
    html.Div([
        dcc.Graph(id='Sent_Hist_exp', figure={}),
    ], style={'display': 'inline-block', 'width': '49%'}),
]),
    
    
])
    @app.callback(
        [Output(component_id='sentiment-table', component_property='data'),
         Output(component_id='Sent_Score', component_property='figure'),
         Output(component_id='Sent_Hist', component_property='figure'),
         Output(component_id='Sent_Hist_exp', component_property='figure')],
        [Input(component_id='Date-picker', component_property='date'),]
        )
    
    
    def update_sent(date):

        real_date=pd.Timestamp(date)
        selected_data=sent_data.loc[sent_data['Date'] <= real_date]
        news=literal_eval(selected_data['News'].iloc[-1])
        new_scores=literal_eval(selected_data['Sent Scores'].iloc[-1])
        ret_data=pd.DataFrame({'News':news,'Scores':new_scores})
        day_score=selected_data['Score'].iloc[-1]
        hist_data=selected_data[['Date','Score','Exponential Average']].iloc[-8:-1]
        #hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=day_score ,
        number=dict(
            prefix="",
            suffix="",
            font=dict(size=80, color="black"),
            ),
            gauge=dict(
                axis=dict(range=[-9, 9]),
                bar=dict(color="black", thickness=0.6),
                steps=[
                    dict(range=[-9, -4], color="red"),
                    dict(range=[-4, 0], color="orange"),
                    dict(range=[0, 4], color="lightgreen"),
                    dict(range=[4, 9], color="green"),
                ],
                
            ),
        ))
        fig4 = go.Figure(data=[go.Bar(
            x=hist_data['Date'],
            y=hist_data['Score'],
            marker=dict(
                color=['green' if score > 0 else 'red' for score in hist_data['Score']],
            )
        )])

        fig4.update_layout(
            title='Previous Week Scores',
            xaxis_title='Date',
            yaxis_title='Cumulative Score',
            bargap=0.2,
            bargroupgap=0.1
        )
        fig5 = go.Figure(data=[go.Bar(
            x=hist_data['Date'],
            y=hist_data['Exponential Average'],
            marker=dict(
                color=['green' if score > 0 else 'red' for score in hist_data['Exponential Average']],
            )
        )])

        fig5.update_layout(
            title='7 Days Exponential Averages',
            xaxis_title='Date',
            yaxis_title='Exponential Average',
            bargap=0.2,
            bargroupgap=0.1
        )


        return  [ret_data.to_dict('records') , fig3 , fig4, fig5]    
    return layout
    
###########################################################################################################

###########################################################################################################
def technical():
    from dash import dash_table
    options = [
    {'label': 'Moving Average (MA)', 'value': 'MA'},
    {'label': 'Relative Strength Index (RSI)', 'value': 'RSI'},
    {'label': 'Bolinger Bands', 'value': 'BB'},
    {'label': 'Moving Average Convergence Divergence (MACD)', 'value': 'MACD'},
    {'label': 'Scholastic Oscillator','value':'SO'},
    {'label': 'Fibonacci Retracement', 'value': 'FR'},
    {'label': 'Williams % R', 'value': 'WR'},
    {'label': 'Momentum', 'value': 'M'},
    {'label': 'Avarage True Range (ATR)', 'value': 'ATR'},
    {'label': 'Commodity Channel Index (CCI)', 'value': 'CCI'},
    {'label': 'Rate of Change (ROC)', 'value': 'ROC'},
    {'label': 'On Balance Volume (OBV)', 'value': 'OBV'},
    ]
    
    second_options = {
    'MA':'MA Window',
    'RSI':' RSI Window',
    'BB': 'BB Window and BB Standard Deviation',
    'MACD': 'MACD Window Short , MACD Window Long and MACD Signal',
    'SO':'SO Window and SO Smooth Window',
    'FR':'FR Swing High and FR Swing Low',
    'WR':'WR Window',
    'M':'M Window',
    'ATR':'ATR Period',
    'CCI':'CCI Period',
    'ROC':'ROC Period',
    'OBV':'OBV Period'

}
    param_no={
        'MA':1,
        'RSI':1,
        'BB':2,
        'MACD':3,
        'SO':2,
        'FR':2,
        'WR':1,
        'M':1,
        'ATR':1,
        'CCI':1,
        'ROC':1,
        'OBV':1
        }
        
        
    checkbox_options = [{'label': option['label'], 'value': option['value']} for option in options]
    div_style = {'display': 'none'}
    layout=html.Div(children=[
        html.H1("Crude Oil WTI Front Month Futures ", style={'text-align': 'center'}),              
        
    html.Div([
        dcc.Checklist(
            id='checkbox-options',
            options=checkbox_options,
            value=[],
            labelStyle={'display': 'block'}
        ),
        html.Label('Indicator Parameters:',id='param-header', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
        dcc.Input(
            id='parameter-input',
            type='text',
            placeholder='Please enter Indicator Parameters',
            style={'width': '100%','font-size': '10px'}
            ),
        
        
    ], style={'display': 'inline-block', 'verticalAlign': 'top','width': '100%'}),
    
      
    
    dcc.Tabs(id="tabs", value='1d', children=[
        dcc.Tab(label='1 Hour', value='1h'),
        dcc.Tab(label='4 Hours', value='4h'),
        dcc.Tab(label='Daily', value='1d'),
        dcc.Tab(label='Weekly', value='1w'),
    ]),
    html.H1("WTI Futures Chart", style={'text-align': 'center'}),
    html.Div( 
        dcc.Graph(id='Candle', figure={}),
        ),
    html.Div( id='MACD',children=[]
        ),
    html.Div( id='cizgi',children=[]
        ),
    html.Div( 
        dcc.Graph(id='Bar', figure={}),
        ),
    html.Div(
        dash_table.DataTable(
            id='tradingrule-table',
            columns=[{'name': 'Technical Trading Rules', 'id': 'Technical Trading Rules'}, {'name': 'Scores', 'id': 'Scores'},{'name': 'Signal', 'id': 'Signal'}],
            data=[],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'Signal', 'filter_query': '{Signal} = "Buy"'},
                    'backgroundColor': 'green',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Signal', 'filter_query': '{Signal} = "Sell"'},
                    'backgroundColor': 'red',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Signal', 'filter_query': '{Signal} = "Neutral"'},
                    'backgroundColor': 'grey',
                    'color': 'white'
                }
            ]
    
            
          
            ),
        ),
    html.Div(
        dash_table.DataTable(
            id='tradingrule-result',
            columns=[{'name': 'Positive', 'id': 'Positive'}, {'name': 'Negative', 'id': 'Negative'},
                     {'name': 'Neutral', 'id': 'Neutral'},{'name': 'Summary', 'id': 'Summary'}],
            data=[],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'Summary', 'filter_query': '{Summary} = "Strong Buy"'},
                    'backgroundColor': 'green',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Summary', 'filter_query': '{Summary} = "Slighly Buy"'},
                    'backgroundColor': '#00FF00',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Summary', 'filter_query': '{Summary} = "Neutral"'},
                    'backgroundColor': 'grey',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Summary', 'filter_query': '{Summary} = "Slightly Sell"'},
                    'backgroundColor': '#FF0000',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Summary', 'filter_query': '{Summary} = "Strong Sell"'},
                    'backgroundColor': 'red',
                    'color': 'white'
                },
                
            ]
    
            
          
            ),
        ),
    html.H1("Backtesting for Technical Trading Rules", style={'text-align': 'center'}),
    html.Div(
        dcc.DatePickerRange(
        id='backtest_picker',
        min_date_allowed=date(2000, 2, 1),
        max_date_allowed=date(2023, 6, 10),
        initial_visible_month=date(2022, 4, 5),
        start_date=date(2021,4,5),
        end_date=date(2022, 7, 23)
        )
        ),
    html.Div([
        dcc.RadioItems(
            id='backtest_radio',
            options=[
        {'label': 'RSI-14', 'value': 'RSI'},
        {'label': 'MACD 12-26-9', 'value': 'MACD'},
        {'label': 'Williams %R', 'value': 'WR'},
        {'label': 'CCI-14', 'value': 'CCI'},
        {'label': 'ATR-14', 'value': 'ATR'},
        {'label': 'ROC-14', 'value': 'ROC'},
        {'label': 'STOCH', 'value': 'STOCH'},
        {'label': 'STOCHRSI', 'value': 'STOCHRSI'},
        {'label': 'ADX-14', 'value': 'ADX'},
        {'label': 'High-Lows-14', 'value': 'HL'},
        {'label': 'Ultimate Oscilator', 'value': 'UO'},
        {'label': 'Bull-Bear-13', 'value': 'BB'},
        {'label': 'MA-5', 'value': 'MA5'},
        {'label': 'MA-10', 'value': 'MA10'},
        {'label': 'MA-20', 'value': 'MA20'},
        {'label': 'MA-50', 'value': 'MA50'},
        {'label': 'MA-100', 'value': 'MA100'},
        {'label': 'MA-200', 'value': 'MA200'},
        ],
            value='RSI',
            style={'display': 'inline-block', 'width': '49%'}
                    ),
        dcc.Graph(id='Test_result', figure={},style={'display': 'inline-block', 'width': '49%'})
        ]
        ),
       
        
      
    
    
])


    @app.callback(
    [Output('parameter-input', 'placeholder'),Output('param-header', 'children'), Output('Candle', 'figure'),Output('Bar', 'figure'),Output('MACD','children'),
     Output('cizgi','children'),Output('tradingrule-table','data'),Output('tradingrule-result','data'),Output('Test_result','figure')],
    [Input('checkbox-options', 'value'), Input('tabs', 'value'),Input('parameter-input','value'),Input('backtest_picker','start_date'),
     Input('backtest_picker','end_date'),Input('backtest_radio','value')]
)
    def render_indicators(selected_options,tabs,parameter_values,sd,ed,value):
        selected_indicators = [option for option in selected_options if option in second_options]
        placeholder = 'Please enter Indicator Parameters : ' + ', '.join(second_options[indicator] for indicator in selected_indicators) + '. Enter parameters for selected indicators separated with comma ( , )'
        print(selected_options,tabs,parameter_values)
        ##############
        cizgiler=['RSI','SO','WR','M','ATR','CCI','ROC']
        sel_ciz=[]
        for z in selected_options:
            if z in cizgiler:
                sel_ciz.append(z)
        ciz_cnt=0
        ciz_graph=None
        print('selected çizgiler',sel_ciz)       
        on_bar_graphs=selected_options
        line_graphs=['MA','BB','FR','OBV']
        macd_graph=None
        print(on_bar_graphs)
        selected_indicators=selected_options
        if str(parameter_values)=='None':
            all_params=[]
        else:
            all_params=str(parameter_values).split(',')
        all_params_list=[]
        for j in all_params:
            all_params_list.append(j)
        print(all_params_list)
        param_locs=[]
        init=0
        for idx in selected_options:
            param_locs.append(init)
            init=init+param_no[idx]
        
        print('init is',init,len(all_params_list))
        ######################################
        if tabs=='1d':
            df_f=df_candle.copy()
            fig_candle = go.Figure(go.Candlestick(
                x=df_f['Date'],
                open=df_f['Open'],
                high=df_f['High'],
                low=df_f['Low'],
                close=df_f['Close']
                ))
            fig_bar=go.Figure()
            fig_bar.add_trace(go.Bar(x=df_f['Date'], y=df_f['Vol.'], name='Volume', marker=dict(color='black')))
            fig_bar.update_layout(
                plot_bgcolor='rgb(255, 255, 255)',  # Beyaz arka plan
                paper_bgcolor='rgb(255, 255, 255)'
            )

                
            
        elif tabs=='1w':
            df_f=dfw.copy()
            fig_candle = go.Figure(go.Candlestick(
                x=df_f['Date'],
                open=df_f['Open'],
                high=df_f['High'],
                low=df_f['Low'],
                close=df_f['Close']
                ))
            fig_bar=go.Figure()
            fig_bar.add_trace(go.Bar(x=df_f['Date'], y=df_f['Vol.'], name='Volume', marker=dict(color='black')))
            fig_bar.update_layout(
                plot_bgcolor='rgb(255, 255, 255)',  # Beyaz arka plan
                paper_bgcolor='rgb(255, 255, 255)'
            )
        elif  tabs=='1h':
            df_f=df_1h.tail(10000).copy()
            fig_candle = go.Figure(go.Candlestick(
                x=df_f['Date'],
                open=df_f['Open'],
                high=df_f['High'],
                low=df_f['Low'],
                close=df_f['Close']
                ))
            fig_bar=go.Figure()
            fig_bar.add_trace(go.Bar(x=df_f['Date'], y=df_f['Vol.'], name='Volume', marker=dict(color='black')))
            fig_bar.update_layout(
                plot_bgcolor='rgb(255, 255, 255)',  # Beyaz arka plan
                paper_bgcolor='rgb(255, 255, 255)'
            )
        elif tabs=='4h':
            df_f=df_4h.tail(10000).copy()
            fig_candle = go.Figure(go.Candlestick(
                x=df_f['Date'],
                open=df_f['Open'],
                high=df_f['High'],
                low=df_f['Low'],
                close=df_f['Close']
                ))
            fig_bar=go.Figure()
            fig_bar.add_trace(go.Bar(x=df_f['Date'], y=df_f['Vol.'], name='Volume', marker=dict(color='black')))
            fig_bar.update_layout(
                plot_bgcolor='rgb(255, 255, 255)',  # Beyaz arka plan
                paper_bgcolor='rgb(255, 255, 255)'
            )
            
            
            
        for ind in range(len(on_bar_graphs)):
            sel_ind=on_bar_graphs[ind]
            if sel_ind=='MA' and len(all_params_list)==init:
                df_f=calculate_ma(df_f, int(all_params_list[param_locs[ind]]))
                fig_candle.add_trace(go.Scatter(
                x=df_f['Date'],
                y=df_f[sel_ind],
                mode='lines',
                name='Moving Average'
            ))
            if sel_ind=='BB' and len(all_params_list)==init:
                df_f=calculate_bollinger_bands(df_f,int(all_params_list[param_locs[ind]]),int(all_params_list[param_locs[ind]+1]))
                fig_candle.add_trace(go.Scatter(
                x=df_f['Date'],
                y=df_f['Upper Band'],
                mode='lines',
                name='Bolinger Upper Band'
            ))
                fig_candle.add_trace(go.Scatter(
                x=df_f['Date'],
                y=df_f['Lower Band'],
                mode='lines',
                name='Bollinger Lower Band'
            ))
            if sel_ind=='FR' and len(all_params_list)==init:
               df_f=calculate_fibonacci_retracement(df_f,int(all_params_list[param_locs[ind]]),int(all_params_list[param_locs[ind]+1])) 
               for idz in range(7):
                   fig_candle.add_trace(go.Scatter(
                   x=df_f['Date'],
                   y=df_f[f'Retracement {idz}'],
                   mode='lines',
                   name=f'Retracement Level {idz}'
                   ))
            if sel_ind=='OBV' and len(all_params_list)==init:
                df_f=calculate_obv(df_f,int(all_params_list[param_locs[ind]]))
                fig_candle.add_trace(go.Scatter(
                x=df_f['Date'],
                y=df_f['OBV'],
                mode='lines',
                name='On Balance Volume'
                ))
                   
            
            if 'MACD' in selected_options and len(all_params_list)==init:
                df_f=calculate_macd(df_f, int(all_params_list[param_locs[ind]]),int(all_params_list[param_locs[ind]+1]), int(all_params_list[param_locs[ind]+2]))
                fig_macd = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

                fig_macd.add_trace(go.Bar(x=df_f['Date'], y=df_f['histogram'], name='MACD Histogram', marker=dict(color='black')), secondary_y=False)
                fig_macd.add_trace(go.Scatter(x=df_f['Date'], y=df_f['macd_line'], mode='lines', name='MACD Line'), secondary_y=True)
                fig_macd.add_trace(go.Scatter(x=df_f['Date'], y=df_f['signal_line'], mode='lines', name='MACD Signal Line'), secondary_y=True)
                
                fig_macd.update_layout(
                    yaxis=dict(title='MACD Histogram', side='left', showgrid=False),
                    yaxis2=dict(title='MACD Line/Signal Line', side='right', showgrid=False),
                    legend=dict(x=0, y=1, traceorder='normal'),
                    plot_bgcolor='rgb(255, 255, 255)',
                    paper_bgcolor='rgb(255, 255, 255)'
                )
                
                macd_graph = dcc.Graph(id='Macd_graph', figure=fig_macd)
                
                macd_graph = dcc.Graph(id='Macd_graph', figure=fig_macd)    
            if on_bar_graphs[ind] in sel_ciz and len(all_params_list)==init:
                if on_bar_graphs[ind] =='RSI':
                    df_f=calculate_rsi(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['rsi'], mode='lines', name='RSI Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['rsi'], mode='lines', name='RSI Line'))
                elif on_bar_graphs[ind] =='SO':
                    df_f=calculate_stochastic_oscillator(df_f, int(all_params_list[param_locs[ind]]),int(all_params_list[param_locs[ind]+1]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['stochastic_k'], mode='lines', name='Stochastic D Line'))
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['stochastic_d'], mode='lines', name='Stochastic K Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['stochastic_k'], mode='lines', name='Stochastic D Line'))
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['stochastic_d'], mode='lines', name='Stochastic K Line'))
                elif on_bar_graphs[ind] =='WR':
                    df_f=calculate_williams_r(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['williams_r'], mode='lines', name='Williams %R Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['williams_r'], mode='lines', name='Williams %R Line'))
                elif on_bar_graphs[ind] =='M':
                    df_f=calculate_momentum(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Momentum'], mode='lines', name='Momentum Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Momentum'], mode='lines', name='Momentum Line'))
                elif on_bar_graphs[ind] =='ATR':
                    df_f=calculate_average_true_range(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['ATR'], mode='lines', name='ATR Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['ATR'], mode='lines', name='ATR Line'))
                elif on_bar_graphs[ind] =='CCI':
                    df_f=calculate_commodity_channel_index(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['CCI'], mode='lines', name='CCI Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['CCI'], mode='lines', name='CCI Line'))
                elif on_bar_graphs[ind] =='ROC':
                    df_f=calculate_rate_of_change(df_f, int(all_params_list[param_locs[ind]]))
                    if ciz_cnt ==0:
                        fig_ciz=go.Figure()
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['ROC'], mode='lines', name='ROC Line'))
                        ciz_cnt=1
                    else:
                        fig_ciz.add_trace(go.Scatter(x=df_f['Date'], y=df_f['ROC'], mode='lines', name='ROC Line'))
                
                ciz_graph=dcc.Graph(id='cizgi_graph',figure=fig_ciz)
                #####################################################################################################################Technical trading rules
        data = {
            'Technical Trading Rules': [None] * 18,
            'Scores': [None] * 18,
            'Signal': [None] * 18
        }
        table_df=pd.DataFrame(data)
        if tabs=='1d':
            df_tr=df_candle.tail(7000).copy()
        elif tabs=='1w':
            df_tr=dfw.copy()
        elif tabs=='4h':
            df_tr=df_4h.tail(7000).copy()
        elif tabs=='1h':
            df_tr=df_1h.tail(7000).copy()
        ###RSI
        df_tr=calculate_rsi(df_tr, 14)
        table_df['Technical Trading Rules'].iloc[0]='RSI (14)'
        table_df['Scores'].iloc[0],table_df['Signal'].iloc[0]=rsi_signal(df_tr)
        ##### MACD
        table_df['Technical Trading Rules'].iloc[1]='MACD (12,26,9)'
        table_df['Scores'].iloc[1],table_df['Signal'].iloc[1]=calculate_macd_signal(df_tr)
        ###### Williams R
        df_tr=calculate_williams_r(df_tr, 14)
        table_df['Technical Trading Rules'].iloc[2]='Williams %R (14,-20,-80)'
        table_df['Scores'].iloc[2],table_df['Signal'].iloc[2]=generate_signals_WR(df_tr)
        ############# CCI
        df_tr=calculate_commodity_channel_index(df_tr,14)
        table_df['Technical Trading Rules'].iloc[3]='CCI (14)'
        table_df['Scores'].iloc[3],table_df['Signal'].iloc[3]=generate_signals_CCI(df_tr)     
        ##########ATR
        df_tr=calculate_average_true_range(df_tr,14)
        table_df['Technical Trading Rules'].iloc[4]='ATR (14)'
        table_df['Scores'].iloc[4],table_df['Signal'].iloc[4]=generate_signals_ATR(df_tr) 
        ############## ROC
        df_tr=calculate_rate_of_change(df_tr,14)
        table_df['Technical Trading Rules'].iloc[5]='ROC (14)'
        table_df['Scores'].iloc[5],table_df['Signal'].iloc[5]=generate_signals_ROC(df_tr) 
        ############## STOCH
        df_tr=calculate_stoch(df_tr)
        table_df['Technical Trading Rules'].iloc[6]='STOCH'
        table_df['Scores'].iloc[6],table_df['Signal'].iloc[6]=generate_signals_stoch(df_tr)         
        ########## STOCHRSI
        df_tr=calculate_stochrsi(df_tr)
        table_df['Technical Trading Rules'].iloc[7]='STOCHRSI'
        table_df['Scores'].iloc[7],table_df['Signal'].iloc[7]=generate_signals_stochrsi(df_tr) 
        ########### ADX
        df_tr=calculate_adx(df_tr)
        table_df['Technical Trading Rules'].iloc[8]='ADX(14)'
        table_df['Scores'].iloc[8],table_df['Signal'].iloc[8]=generate_signals_ADX(df_tr)     
        ########## Highs-Lows
        df_tr=calculate_highs_lows(df_tr)
        table_df['Technical Trading Rules'].iloc[9]='Highs / Lows (14)'
        table_df['Scores'].iloc[9],table_df['Signal'].iloc[9]=generate_signals_highlow(df_tr) 
        ##########Ultimate Oscillator
        df_tr=calculate_ultimate_oscillator(df_tr)
        table_df['Technical Trading Rules'].iloc[10]='Ultimate Oscillator'
        table_df['Scores'].iloc[10],table_df['Signal'].iloc[10]=generate_signals_uo(df_tr) 
        ########## Bull Bear
        df_tr=calculate_ema_bb(df_tr)
        df_tr=calculate_bull_bear_power(df_tr)
        table_df['Technical Trading Rules'].iloc[11]='Bull Bear (13)'
        table_df['Scores'].iloc[11],table_df['Signal'].iloc[11]=generate_signals_bbp(df_tr) 
        ######## MA5
        df_tr=calculate_ma_tech(df_tr,5)
        table_df['Technical Trading Rules'].iloc[12]='MA(5)'
        table_df['Scores'].iloc[12],table_df['Signal'].iloc[12]=generate_signals_mas(df_tr,5) 
        ###### MA 10
        df_tr=calculate_ma_tech(df_tr,10)
        table_df['Technical Trading Rules'].iloc[13]='MA(10)'
        table_df['Scores'].iloc[13],table_df['Signal'].iloc[13]=generate_signals_mas(df_tr,10)        
        ######MA 20
        df_tr=calculate_ma_tech(df_tr,20)
        table_df['Technical Trading Rules'].iloc[14]='MA(20)'
        table_df['Scores'].iloc[14],table_df['Signal'].iloc[14]=generate_signals_mas(df_tr,20)  
        #########MA 50
        df_tr=calculate_ma_tech(df_tr,50)
        table_df['Technical Trading Rules'].iloc[15]='MA(50)'
        table_df['Scores'].iloc[15],table_df['Signal'].iloc[15]=generate_signals_mas(df_tr,50)  
        #########MA 100
        df_tr=calculate_ma_tech(df_tr,100)
        table_df['Technical Trading Rules'].iloc[16]='MA(100)'
        table_df['Scores'].iloc[16],table_df['Signal'].iloc[16]=generate_signals_mas(df_tr,100)  
        ########## MA 200
        df_tr=calculate_ma_tech(df_tr,200)
        table_df['Technical Trading Rules'].iloc[17]='MA(200)'
        table_df['Scores'].iloc[17],table_df['Signal'].iloc[17]=generate_signals_mas(df_tr,200)  
        
        data2 = {
            'Positive': [None] * 1,
            'Negative': [None] * 1,
            'Neutral': [None] * 1,
            'Summary':[None]*1
        }
        table_res=pd.DataFrame(data2)
        pos_num=len(table_df.loc[table_df['Signal']=='Buy'])
        neg_num=len(table_df.loc[table_df['Signal']=='Sell'])
        neut_num=len(table_df.loc[table_df['Signal']=='Neutral'])
        table_res['Positive'].iloc[0]=pos_num
        table_res['Negative'].iloc[0]=neg_num
        table_res['Neutral'].iloc[0]=neut_num
        if pos_num-neg_num >2 and pos_num-neg_num <5:
            summary='Slightly Buy'
        elif pos_num-neg_num >=5:
            summary='Strong Buy'
        elif pos_num-neg_num <=2 and pos_num-neg_num >= -2:
            summary='Neutral'
        elif pos_num-neg_num <-2 and pos_num-neg_num > -5:
            summary='Slightly Sell'
        elif pos_num-neg_num <= -5:
            summary='Strong Sell'
        table_res['Summary'].iloc[0]=summary
        
        ################################################################################################### BAcktest

        df_bck=df_tr[(df_tr['Date']>=sd)&(df_tr['Date']<=ed)].copy()
        if value=='RSI':
            df_bck=calculate_rsi(df_bck, 14)
            df_bck['Buy']=0
            df_bck['Sell']=0
            df_bck['Cost L']=0
            df_bck['Cost S']=0
            df_bck.loc[df_bck['rsi']>=70,'Buy']=1
            df_bck.loc[df_bck['rsi']<=30,'Sell']=1
            pos=0
            n_pos=0
            amt2=0
            amt=0
            l=len(df_bck)
            cnt=0
            for idx,row in df_bck.iterrows():
                if row['Buy']==1 and pos==0 and amt==0:
                    amt=1000/row['Close']
                    df_bck.loc[idx, 'Cost L'] = -1000
                    pos=1
                if row['Sell']==1 and pos==1:                    
                    df_bck.loc[idx, 'Cost L'] =row['Close']*amt
                    pos=0
                    amt=0
                if pos==0 and row['Sell']==1 and amt2==0:
                    amt2=1000/row['Close']
                    df_bck.loc[idx, 'Cost S'] = 1000
                    n_pos=1
                if n_pos==1 and row['Buy']==1:
                    df_bck.loc[idx, 'Cost S'] =-row['Close']*amt2
                    n_pos=0
                    amt2=0
                if cnt == l - 1 and n_pos==1:
                    df_bck.loc[idx, 'Cost S'] =-row['Close']*amt2
                    n_pos=0
                    amt2=0
                if cnt == l - 1 and pos==1:
                    df_bck.loc[idx, 'Cost S'] =row['Close']*amt
                    pos=0
                    amt=0
                cnt+=1
            profit1=sum(df_bck['Cost L'])
            profit2=sum(df_bck['Cost S'])
            profit=profit1+profit2
            if profit == 0:
                profit=1000
            else:
                profit=profit+1000
        print(profit)        
        fig_prf = go.Figure()
        fig_prf.add_trace(go.Indicator(
        mode = "number+delta",
        value = profit,
        delta = {'reference': 1000, 'relative': True},
        domain = {'x': [0, 0.5], 'y': [0.2, 1]}))
    

        return placeholder,html.Label(placeholder, id='param-header', style={'font-weight': 'bold', 'margin-bottom': '5px'}),fig_candle,fig_bar,macd_graph,ciz_graph,table_df.to_dict('records'),table_res.to_dict('records'),fig_prf
    

        
    return layout
########################################################################################################♣3# Trading Rule Functions
def rsi_signal(df,rsi_window=14, upper_threshold=70, lower_threshold=30):
    value=df['rsi'].iloc[-1]
    if value >= upper_threshold:
        signal='Buy'
    elif value <= lower_threshold:
        signal='Sell'
    else:
        signal='Neutral'
    return round(value,3),signal
def calculate_macd_signal(df, n_fast=12, n_slow=26, n_signal=9):

    df['EMA_fast'] = df['Close'].ewm(span=n_fast, min_periods=n_fast).mean()
    df['EMA_slow'] = df['Close'].ewm(span=n_slow, min_periods=n_slow).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal_MACD'] = df['MACD'].ewm(span=n_signal, min_periods=n_signal).mean()
    df['Histogram'] = df['MACD'] - df['Signal_MACD']
    
    # Sinyal üretimi
    df['Signal_Signal'] = 0
    df.loc[(df['Histogram'] > 0) & (df['Histogram'].shift(1) < 0), 'Signal_Signal'] = 1
    df.loc[(df['Histogram'] < 0) & (df['Histogram'].shift(1) > 0), 'Signal_Signal'] = -1

    sign = df['Signal_Signal'].iloc[-1]
    value=df['MACD'].iloc[-1]
    
    df.drop(['EMA_fast', 'EMA_slow', 'Histogram', 'Signal_Signal'], axis=1, inplace=True)
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'
    
    return  round(value,3),signal

  
def generate_signals_WR(data, upper_threshold=-20, lower_threshold=-80):
    data['Signal_WR'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['williams_r'] >= upper_threshold, 'Signal_WR'] = -1  # Sell signal (Williams %R above upper threshold)
    data.loc[data['williams_r'] <= lower_threshold, 'Signal_WR'] = 1  # Buy signal (Williams %R below lower threshold)

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_WR'] = data['Signal_WR'].diff()
    value=data['williams_r'].iloc[-1]
    sign=data['Signal_WR'].iloc[-1]
    if sign ==1 :
        signal='Buy'
    elif sign==-1 :
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal    
def generate_signals_CCI(data, upper_threshold=100, lower_threshold=-100):
    data['Signal_CCI'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['CCI'] >= upper_threshold, 'Signal_CCI'] = -1  # Sell signal (CCI above upper threshold)
    data.loc[data['CCI'] <= lower_threshold, 'Signal_CCI'] = 1  # Buy signal (CCI below lower threshold)

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_CCI'] = data['Signal_CCI'].diff()
    sign=data['Signal_CCI'].iloc[-1]
    value=data['CCI'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal 
def generate_signals_ATR(data, atr_multiplier=2):
    data['Signal_ATR'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    atr = data['ATR'].iloc[-1]  # Son ATR değeri

    # Generate trading signals
    data.loc[data['Close'] > data['Close'].shift(1) + atr_multiplier * atr, 'Signal_ATR'] = 1  # Long position
    data.loc[data['Close'] < data['Close'].shift(1) - atr_multiplier * atr, 'Signal_ATR'] = -1  # Short position

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_ATR'] = data['Signal_ATR'].diff()
    sign=data['Signal_ATR'].iloc[-1]
    value=data['ATR'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal 
def generate_signals_ROC(data, threshold=5):
    data['Signal_ROC'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['ROC'] > threshold, 'Signal_ROC'] = -1  # Sell signal (ROC above threshold)
    data.loc[data['ROC'] < -threshold, 'Signal_ROC'] = 1  # Buy signal (ROC below threshold)

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_ROC'] = data['Signal_ROC'].diff()
    sign=data['Signal_ROC'].iloc[-1]
    value=data['ROC'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal 
def calculate_stoch(data, window=14, k_window=3, d_window=3):
    data['Lowest_Low'] = data['Low'].rolling(window=window).min()
    data['Highest_High'] = data['High'].rolling(window=window).max()
    data['%K'] = (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']) * 100
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    return data

def generate_signals_stoch(data, k_threshold=80, d_threshold=80):
    data['Signal_Stoch'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[(data['%K'] > k_threshold) & (data['%D'] > d_threshold), 'Signal_Stoch'] = -1  # Sell signal (both %K and %D above thresholds)
    data.loc[(data['%K'] < k_threshold) & (data['%D'] < d_threshold), 'Signal_Stoch'] = 1  # Buy signal (both %K and %D below thresholds)

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_Stoch'] = data['Signal_Stoch'].diff()
    sign=data['Signal_Stoch'].iloc[-1]
    value=data['%K'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal
####Stoch RSI
def calculate_stochrsi(data, period=14, rsi_period=14, stoch_period=3):
    data['RSI'] = calculate_rsi_stoch(data['Close'], rsi_period)
    data['StochRSI'] = (data['RSI'] - data['RSI'].rolling(period).min()) / (data['RSI'].rolling(period).max() - data['RSI'].rolling(period).min())
    data['StochRSI_K'] = data['StochRSI'].rolling(stoch_period).mean()
    return data

def calculate_rsi_stoch(prices, period):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period, len(prices)):
        delta = deltas[i - 1]  # Current price change
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def generate_signals_stochrsi(data, oversold_threshold=0.2, overbought_threshold=0.8):
    data['Signal_stochrsi'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['StochRSI_K'] < oversold_threshold, 'Signal_stochrsi'] = 1  # Long position
    data.loc[data['StochRSI_K'] > overbought_threshold, 'Signal_stochrsi'] = -1  # Short position

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_stochrsi'] = data['Signal_stochrsi'].diff()
    sign=data['Signal_stochrsi'].iloc[-1]
    value=data['StochRSI_K'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal
####ADX
def calculate_adx(data, period=14):
    data['TR'] = calculate_true_range(data)
    data['DM_plus'], data['DM_minus'] = calculate_directional_movement(data)
    data['DI_plus'] = 100 * (data['DM_plus'].rolling(period).sum() / data['TR'].rolling(period).sum())
    data['DI_minus'] = 100 * (data['DM_minus'].rolling(period).sum() / data['TR'].rolling(period).sum())
    data['DX'] = 100 * np.abs(data['DI_plus'] - data['DI_minus']) / (data['DI_plus'] + data['DI_minus'])
    data['ADX'] = data['DX'].rolling(period).mean()
    return data

def calculate_true_range(data):
    high_minus_low = data['High'] - data['Low']
    high_minus_prev_close = np.abs(data['High'] - data['Close'].shift(1))
    low_minus_prev_close = np.abs(data['Low'] - data['Close'].shift(1))
    true_range = np.max([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=0)
    return true_range

def calculate_directional_movement(data):
    move_up = data['High'] - data['High'].shift(1)
    move_down = data['Low'].shift(1) - data['Low']
    dm_plus = np.where((move_up > 0) & (move_up > move_down), move_up, 0)
    dm_minus = np.where((move_down > 0) & (move_down > move_up), move_down, 0)
    return dm_plus, dm_minus

def generate_signals_ADX(data, adx_threshold=25, di_threshold=20):
    data['Signal_ADX'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[(data['ADX'] > adx_threshold) & (data['DI_plus'] > data['DI_minus']) & (data['DI_plus'] > di_threshold), 'Signal_ADX'] = 1  # Long position
    data.loc[(data['ADX'] > adx_threshold) & (data['DI_minus'] > data['DI_plus']) & (data['DI_minus'] > di_threshold), 'Signal_ADX'] = -1  # Short position

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_ADX'] = data['Signal_ADX'].diff()
    sign=data['Signal_ADX'].iloc[-1]
    value=data['ADX'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal
###########high low
def calculate_highs_lows(data, period=14):
    data['HH'] = data['High'].rolling(period).max()
    data['LL'] = data['Low'].rolling(period).min()
    data['Highs_Lows'] = data['HH'] - data['LL']
    return data

def generate_signals_highlow(data, threshold=2):
    data['Signal_HL'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['Highs_Lows'] > threshold, 'Signal_HL'] = 1  # Long position
    data.loc[data['Highs_Lows'] < -threshold, 'Signal_HL'] = -1  # Short position

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_HL'] = data['Signal_HL'].diff()
    sign=data['Signal_HL'].iloc[-1]
    value=data['Highs_Lows'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal

#####################################ultimate oscilator
def calculate_ultimate_oscillator(data, period1=7, period2=14, period3=28):
    data['BP'] = data['Close'] - data['Low'].rolling(period1).min()
    data['TR'] = data['High'].rolling(period1).max() - data['Low'].rolling(period1).min()

    data['BP_sum1'] = data['BP'].rolling(period1).sum()
    data['TR_sum1'] = data['TR'].rolling(period1).sum()
    data['BP_sum2'] = data['BP'].rolling(period2).sum()
    data['TR_sum2'] = data['TR'].rolling(period2).sum()
    data['BP_sum3'] = data['BP'].rolling(period3).sum()
    data['TR_sum3'] = data['TR'].rolling(period3).sum()

    data['UO'] = ((data['BP_sum1'] / data['TR_sum1']) * 4 + (data['BP_sum2'] / data['TR_sum2']) * 2 + (data['BP_sum3'] / data['TR_sum3'])) / 7
    return data

def generate_signals_uo(data, threshold1=30, threshold2=70):
    data['Signal_UO'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['UO'] > threshold2, 'Signal_UO'] = 1  # Long position
    data.loc[data['UO'] < threshold1, 'Signal_UO'] = -1  # Short position

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_UO'] = data['Signal_UO'].diff()
    sign=data['Signal_UO'].iloc[-1]
    value=data['UO'].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal
##########Bullbear
def calculate_ema_bb(data, period=5):
    data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data
def calculate_bull_bear_power(data, period=13):
    data['BullPower'] = data['High'] - data['EMA'].rolling(period).mean()
    data['BearPower'] = data['Low'] - data['EMA'].rolling(period).mean()
    return data

def generate_signals_bbp(data, threshold=0):
    data['Signal_BBP'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['BullPower'] > threshold, 'Signal_BBP'] = 1  # Bullish signal
    data.loc[data['BearPower'] < -threshold, 'Signal_BBP'] = -1  # Bearish signal

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position_BBP'] = data['Signal_BBP'].diff()
    sign=data['Signal_BBP'].iloc[-1]
    value=[round(data['BullPower'].iloc[-1],3), ' / ',round(data['BearPower'].iloc[-1],3)]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return value,signal
##########MA5
def calculate_ma_tech(data, period):
    data['MA'+str(period)] = data['Close'].rolling(period).mean()
    return data

def generate_signals_mas(data,period):
    data['Signal_MA'+str(period)] = 0  # 0: no signal, 1: buy signal, -1: sell signal

    # Generate trading signals
    data.loc[data['Close'] > data['MA'+str(period)], 'Signal_MA'+str(period)] = 1  # Bullish signal
    data.loc[data['Close'] < data['MA'+str(period)], 'Signal_MA'+str(period)] = -1  # Bearish signal

    # Calculate the positions (1 for long, -1 for short, 0 for neutral)
    data['Position'+str(period)] = data['Signal_MA'+str(period)].diff()
    sign=data['Signal_MA'+str(period)].iloc[-1]
    value=data['MA'+str(period)].iloc[-1]
    if sign ==1:
        signal='Buy'
    elif sign==-1:
        signal='Sell'
    else:
        signal='Neutral'

    return round(value,3),signal
########################################################
prd=prediction()
sent_screen=sentiment()
#tech=timeseries()
tech=technical()

model_pred=predictors()
# styling the sidebar

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

    

# Tabs and Tab contents
tabs = dcc.Tabs(id="tabs", value='tab-1', children=[
    dcc.Tab(label='Price Forecasting', value='tab-1'),
    dcc.Tab(label='Model Predictors', value='tab-2')
])



tab_content = html.Div(id='tab-content')


sidebar = html.Div(
    [
        html.H3("Welcome to Crude Oil Trading DSS", className="display-4", style={'fontSize': '30px'}),
        html.Hr(),
        html.P(
            "Make your trading decisions based on WTI price forecasts,technical analysis results, and the impact of news flows.",
            className="lead", style={'fontSize': '16px'}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Fundamental Analysis", href="/", active="exact"),
                dbc.NavLink("Technical Analysis", href="/page-3", active="exact"),
                dbc.NavLink("Sentiment Analysis", href="/page-4", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[tabs, tab_content], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])





@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value")]
)
def render_tab_content(tab):
    if tab == 'tab-1':
        return prd
    elif tab == 'tab-2':
        return model_pred

    return html.P("This page is under construction...")





@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([sidebar, tabs, tab_content])

    elif pathname == "/page-3":
        return tech

    elif pathname == "/page-4":
        return sent_screen

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )






































# =============================================================================
    
if __name__ == '__main__':
    app.run_server(debug=False)