import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
# from datetime import  timedelta
import datetime
import streamlit as st
import plotly.graph_objects as go
import numpy as np


frequency = st.sidebar.number_input(f'Как  редко перекладывать активы', value=1)

option = st.sidebar.selectbox(
    'Выберите валюту, в которой будет изображена динамика активов',
    ('KUSD','KRUR',  'BTC', 'GOLD')
)
portf = st.sidebar.selectbox(
    'Выберите портфель',
    (2, 1)
)

start_date = str(st.sidebar.date_input('Выберите начальную дату', pd.to_datetime("2023-01-01")))
end_date = str(st.sidebar.date_input('Выберите конечную дату', pd.to_datetime("2024-02-15")))
##################################################################################################################################
# def download(frequency = 1):
#             # data  = pd.DataFrame(pd.read_csv('combined_data_full_history.csv')).set_index('Date')
#             # data_btc_usd = data.loc[start_date:end_date, 'High_BTC']
#             # data_rub_USD = data.loc[start_date:end_date, 'High_USDRUB']
#             conversion_factor = 31.1035


#             data_rub_USD = yf.download('USDRUB=X', start=start_date, end=end_date)['High']
#             data_btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['High']
#             gold_data = yf.download('GC=F', start = start_date, end = end_date)['High']
#             gold_usd_gram = gold_data / conversion_factor

#             rub_usd_btc = {}
#             count = 0
#             for i in data_rub_USD.index:
#                 count += 1
#                 try:
#                     if count%frequency == 0:
#                         rub_usd_btc[i] =   [1, data_rub_USD[i], data_btc_usd.loc[i]*data_rub_USD[i],gold_usd_gram[i]*data_rub_USD[i] ]
#                 except:
#                     continue


#             x = list(rub_usd_btc.values())
#             date = list(rub_usd_btc.keys())
#             st.write()
#             def dataframe(dfs):
#                 data_dfs = []
#                 for index in range(len(dfs)):
#                     d = [1000 for i in range(len(dfs[0]))]
                    
#                     data = { 'KRUR': d,'KUSD': d, 'BTC': d, 'GOLD':d}
#                     df = pd.DataFrame(data)
#                     df = df.set_index(df.columns)
#                     df = df.rename_axis(date[index])
#                     df.iloc[0,:] = dfs[index]
#                     for j in range(len(df.columns)):
#                         for i in range(j,len(df.columns)):
#                             if i==j:
#                                 df.iloc[i,j] = 1
#                                 continue
#                             if j<i:
#                                 df.iloc[j,i] = df.iloc[0,i]/df.iloc[0,j]
#                             df.iloc[i,j] = df.iloc[j,j]/df.iloc[j,i]
#                     data_dfs.append(df)
#                 return data_dfs

#             dfs = dataframe(x)
#             return dfs,date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram









# def fill_missing_dates(df):
#     if df.index.isnull().any():
#         return df  # Если есть NaT в индексе, просто возвращаем исходный DataFrame
#     else:
#         all_dates = pd.date_range(start=df.index.min(), end=df.index.max())
#         return df.reindex(all_dates).ffill()  # Заполняем пропущенные даты и значения предыдущими данными


# def download(start_date, end_date, frequency=1):
#     conversion_factor = 31.1035

#     data_rub_USD = yf.download('USDRUB=X', start=start_date, end=end_date)['High']
#     data_btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['High']
#     gold_data = yf.download('GC=F', start=start_date, end=end_date)['High']
#     gold_usd_gram = gold_data / conversion_factor

#     # Создаем DataFrame из полученных данных
#     rub_usd_btc = {}
#     count = 0
#     for i in data_rub_USD.index:
#         count += 1
#         try:
#             if count % frequency == 0:
#                 rub_usd_btc[i] = [1, data_rub_USD[i], data_btc_usd.loc[i] * data_rub_USD[i],
#                                   gold_usd_gram[i] * data_rub_USD[i]]
#         except:
#             continue

#     x = list(rub_usd_btc.values())
#     date = list(rub_usd_btc.keys())

#     def dataframe(dfs):
#         data_dfs = []
#         for index in range(len(dfs)):
#             d = [1000 for i in range(len(dfs[0]))]

#             data = {'KRUR': d, 'KUSD': d, 'BTC': d, 'GOLD': d}
#             df = pd.DataFrame(data)
#             df = df.set_index(df.columns)
#             df = df.rename_axis(date[index])
#             df.iloc[0, :] = dfs[index]
#             for j in range(len(df.columns)):
#                 for i in range(j, len(df.columns)):
#                     if i == j:
#                         df.iloc[i, j] = 1
#                         continue
#                     if j < i:
#                         df.iloc[j, i] = df.iloc[0, i] / df.iloc[0, j]
#                     df.iloc[i, j] = df.iloc[j, j] / df.iloc[j, i]
#             data_dfs.append(df)
#         return data_dfs

#     dfs = dataframe(x)

#     # Заполняем пропущенные даты
#     data_rub_USD_filled = fill_missing_dates(data_rub_USD)
#     data_btc_usd_filled = fill_missing_dates(data_btc_usd)
#     gold_data_filled = fill_missing_dates(gold_data)
#     gold_usd_gram_filled = fill_missing_dates(gold_usd_gram)

#     return dfs, date, data_rub_USD_filled, data_btc_usd_filled, gold_data_filled, gold_usd_gram_filled






def fill_missing_dates(df):
    if df.index.isnull().any():
        return df  # Если есть NaT в индексе, просто возвращаем исходный DataFrame
    else:
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max())
        return df.reindex(all_dates).ffill()  # Заполняем пропущенные даты и значения предыдущими данными

def download(start_date, end_date, frequency=1, filename='data.pkl', update=True):
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2024, 2, 2)
    delta = datetime.timedelta(days=1)

    dates_list = []
    current_date = start_date
    while current_date < end_date:
        dates_list.append(current_date.strftime('%Y-%m-%d %H:%M:%S'))
        current_date += delta
    date = dates_list
    if not update:  # Если update=False, пытаемся загрузить данные из файла
        try:
            # Загружаем данные из файла
            dfs, date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram = pd.read_pickle(filename)
            print("Данные загружены из файла.")
            return dfs, date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram
        except FileNotFoundError:
            print("Файл не найден. Будут загружены новые данные.")

    conversion_factor = 31.1035

    data_rub_USD = yf.download('USDRUB=X', start=start_date, end=end_date)['Open']
    data_btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['Open']
    gold_data = yf.download('GC=F', start=start_date, end=end_date)['Open']
    gold_usd_gram = gold_data / conversion_factor


    # Заполняем пропущенные даты
    data_rub_USD = fill_missing_dates(data_rub_USD)
    data_btc_usd = fill_missing_dates(data_btc_usd)
    gold_data = fill_missing_dates(gold_data)
    gold_usd_gram = fill_missing_dates(gold_usd_gram)
    

    rub_usd_btc = {}
    count = 0
    for i in pd.to_datetime(date):
        count += 1
        try:
            if count % frequency == 0:
                rub_usd_btc[pd.to_datetime(i)] = [1, data_rub_USD[pd.to_datetime(i)], data_btc_usd.loc[pd.to_datetime(i)] * data_rub_USD[pd.to_datetime(i)],
                                    gold_usd_gram[pd.to_datetime(i)] * data_rub_USD[pd.to_datetime(i)]]
        except:
            continue

    x = list(rub_usd_btc.values())
    date = list(rub_usd_btc.keys())

    def dataframe(dfs):
        data_dfs = []
        for index in range(len(dfs)):
            d = [1000 for i in range(len(dfs[0]))]

            data = {'KRUR': d, 'KUSD': d, 'BTC': d, 'GOLD': d}
            df = pd.DataFrame(data)
            df = df.set_index(df.columns)
            df = df.rename_axis(date[index])
            df.iloc[0, :] = dfs[index]
            for j in range(len(df.columns)):
                for i in range(j, len(df.columns)):
                    if i == j:
                        df.iloc[i, j] = 1
                        continue
                    if j < i:
                        df.iloc[j, i] = df.iloc[0, i] / df.iloc[0, j]
                    df.iloc[i, j] = df.iloc[j, j] / df.iloc[j, i]
            data_dfs.append(df)
        return data_dfs

    dfs = dataframe(x)


    # Сохраняем данные в файл
    data_to_save = (dfs, date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram)
    pd.to_pickle(data_to_save, filename)
    # st.write('keys', rub_usd_btc.keys())
    return dfs, date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram


##################################################################################################################################





# def down( start_date='2023-01-01', end_date='2024-01-31'):
#     # Загрузка данных с использованием yfinance
#     rub_usd_data = yf.download('USDRUB=X', start=start_date, end=end_date)['High']
#     btc_usd_data = yf.download('BTC-USD', start=start_date, end=end_date)['High']
#     return st.write(rub_usd_data)

def down_bd(portf):
         # Замените значения переменных на свои
    dbname = "app_invest"
    user = "postgres"
    password = "invest"
    host = "localhost"
    port = "5432"

    # Создание подключения
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

    # Создание курсора
    cursor = conn.cursor()

    cursor.execute("SELECT name_operations, active, value, date FROM operations where id_port = {} order by date".format(portf))
    rows = cursor.fetchall()
    history = {}

    for row in rows:
        active = row[1]
        datetime_str = row[3].strftime('%Y-%m-%d %H:%M:%S')  # предполагается, что date - это объект datetime

        if active not in history:
            history[active] = {}

        if datetime_str not in history[active]:
            history[active][datetime_str] = {}

        operation_type = row[0]
        value = row[2]

        if operation_type not in history[active][datetime_str]:
            history[active][datetime_str][operation_type] = value
        else:
            history[active][datetime_str][operation_type] += value
        cursor.close()
        conn.close()
    cur_port = {'KRUR':[0],  'KUSD':[0] , 'BTC':[0], 'GOLD':[0]}
    for i in cur_port:
         if i not in history:
            history[i] = {date[0]:{'add':0}}
    return history



def MAX_MIN_bd(currency, dfs, date, portf, fig):
            global date_port 

            date_port = pd.DataFrame({'KRUR':[0],  'KUSD':[0] , 'BTC':[0], 'GOLD':[0]},index=date)
            color = {'KRUR': 'red',  'KUSD':'blue' , 'BTC':'orange', 'GOLD':'yellow'}
            history = down_bd(portf)
            cur_port = {'KRUR':[0],  'KUSD':[0] , 'BTC':[0], 'GOLD':[0]}

            pre_sum, sum = 0, 0 
            for i in range(len(date)-1):
                    const = 0
                    for k in list(history.keys()):
                        if  date[i] in history[k].keys():

                            if k not in cur_port.keys():
                                cur_port[k] = [0]
                            if 'add' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] + history[k][date[i]]['add'])
                                date_port.loc[date[i], k] = cur_port[k][-1]



                                fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                                            mode='lines',
                                                            line=dict(color=color[k], dash='dash'),
                                                            name = k,
                                                            text='''+RUB:{},<br>+USD:{},<br>+BTC:{},<BR>+GOLD:{}'''.format(round(history[k][date[i]]['add']*dfs[i].loc['KRUR', k],2),
                                                                                                                                    round(history[k][date[i]]['add']*dfs[i].loc['KUSD', k],2), 
                                                                                                                                    round(history[k][date[i]]['add']*dfs[i].loc['BTC', k],5),
                                                                                                                                    round(history[k][date[i]]['add']*dfs[i].loc['GOLD', k],5),
                                                                                                                                    ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=1
                                                            )

                            if 'withdraw' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] - history[k][date[i]]['withdraw'])
                                date_port.loc[date[i], k] = cur_port[k][-1]

                                fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                                            mode='lines',
                                                            line=dict(color=color[k], dash='dash'),
                                                            name = k,
                                                            text='''-RUB:{},<br>-USD:{},<br>-BTC:{},<BR>-GOLD:{}'''.format(round( history[k][date[i]]['withdraw']*dfs[i].loc['KRUR', k],2),
                                                                                                                                    round( history[k][date[i]]['withdraw']*dfs[i].loc['KUSD', k],2), 
                                                                                                                                    round( history[k][date[i]]['withdraw']*dfs[i].loc['BTC', k],5),
                                                                                                                                    round( history[k][date[i]]['withdraw']*dfs[i].loc['GOLD', k],5),
                                                                                                                                    ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=1
                                                            )
                            for j in list(history.keys()):
                                if j in list(history[k][date[i]].keys()):
                                    if j not in cur_port:
                                            cur_port[j] = [0]
                                    cur_port[k].append(cur_port[k][-1] - history[k][date[i]][j])
                                    date_port.loc[date[i], k] = cur_port[k][-1]

                                    fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                                            mode='lines',
                                                            line=dict(color=color[k], dash='dash'),
                                                            name = k,
                                                            text='''-RUB:{},<br>-USD:{},<br>-BTC:{},<BR>-GOLD:{}'''.format(round(history[k][date[i]][j]*dfs[i].loc['KRUR', k],2),
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['KUSD', k],2), 
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['BTC', k],5),
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['GOLD', k],5),),

                                                            showlegend = False,),
                                                            row=1,  
                                                            col=1
                                                            )

                                    cur_port[j].append(cur_port[j][-1] + history[k][date[i]][j]*dfs[i].loc[j, k])
                                    date_port.loc[date[i], j] = cur_port[j][-1]

                                    fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [cur_port[j][-2]* dfs[i].loc[currency, j], cur_port[j][-1]* dfs[i].loc[currency, j]],
                                                            mode='lines',
                                                            line=dict(color=color[k], dash='dash'),
                                                            name = k,
                                                            text='''+RUB:{},<br>+USD:{},<br>+BTC:{},<BR>+GOLD:{} '''.format(round(history[k][date[i]][j]*dfs[i].loc['KRUR', k],2),
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['BTC', k],5),
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['KUSD', k],2), 
                                                                                                                                    round(history[k][date[i]][j]*dfs[i].loc['GOLD', k],5),),

                                                            showlegend = False,),
                                                            row=1,  
                                                            col=1
                                                            )
                            const +=1
                    if const == 0 and i>0:
                        for k in list(cur_port.keys()):
                            if cur_port[k][-1]!=0:
                                date_port.loc[date[i], k] = cur_port[k][-1]

                                fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                            y= [cur_port[k][-1]* dfs[i-1].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                                                mode='lines',
                                                                line=dict(color=color[k]),
                                                                name = k,
                                                                text='''RUB:{},<br>USD:{},<br>BTC:{},<BR>GOLD:{}'''.format(round(cur_port[k][-1]*dfs[i].loc['KRUR', k],2),
                                                                                                                                        round(cur_port[k][-1]*dfs[i].loc['KUSD', k],2), 
                                                                                                                                        round(cur_port[k][-1]*dfs[i].loc['BTC', k],5),
                                                                                                                                        round(cur_port[k][-1]*dfs[i].loc['GOLD', k],5),),

                                                                showlegend = False,),
                                                                row=1,  
                                                                col=1
                                                                )
                        


                    sum = 0                                        
                    for k in list(cur_port.keys()):
                        sum += cur_port[k][-1]*dfs[i].loc[currency, k]
                            #тут можжно добавить условие если суммы разные без учетов курса за прошлый и этот промеж то рисуем между этими точкми пунктир в эту дату!или сравнивать длину cur_port
                    # capital.append(sum)


                    if const != 0 and i>0:#ситуация когда были изменения капитала за счет перекладываний!
                            
                        


                        #############################################################################считаем статистики для дат в которых были изменения!
                        # volatility()


                        #############################################################################
                        # for k in list(history.keys()):
                        #         if j in list(history[k][date[i]].keys())

                        #     st.write(pre_sum, sum)
                            fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [pre_sum , sum],
                                                            mode='lines',
                                                            line=dict(color='white', dash = 'dash'),
                                                            name = k,
                                                            text='''RUB:{},<br>USD:{},<br>BTC:{},<BR>GOLD:{}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),
                                                                                                                                    round(sum*dfs[i].loc['GOLD', currency],5),
                                                                                                                                    ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=2)

                            fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum , pre_sum ],
                                                        mode='lines',
                                                        line=dict(color='white'),
                                                        name = "Капитал",
                                                            text='''RUB:{},<br>USD:{},<br>BTC:{},<BR>GOLD:{}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),
                                                                                                                                    round(sum*dfs[i].loc['GOLD', currency],5),
                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    else:
                        fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum, sum ],
                                                        mode='lines',
                                                        line=dict(color='white'),
                                                        name = 'Капитал',
                                                            text='''RUB:{},<br>USD:{},<br>BTC:{},<BR>GOLD:{}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),
                                                                                                                                    round(sum*dfs[i].loc['GOLD', currency],5),
                                                                                                                                    
                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    pre_sum = sum
                        
                        
                    for k in list(cur_port.keys()):
                        if cur_port[k][-1]!=0:
                            date_port.loc[date[i], k] = cur_port[k][-1]


            #         if date[i-1] in history[k].keys():
            #             list_f = pd.DataFrame(dfs[i].loc[:,list(cur_port.keys())])
            #             list_s = pd.DataFrame(dfs[i + 1].loc[:,list(cur_port.keys())])
            #             max_values.append(1+(list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
            #             max = func((list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
            #             data1 = list_f.index.name
            #             data2 = list_s.index.name
            #             max_index = func(max_values[-1])

            #             max_index = funcx(max_index)
            #             st.write(max_values[-1], 'max',  max_index)

            #             rub_value = max_values[-1][max_index][0]
            #             usd_value = max_values[-1][max_index][1]
            #             btc_value = max_values[-1][max_index][2]
            #             st.write(rub_value, usd_value,  btc_value)

            #             capital.append(capital[-1]* (max_values[-1][max_index][currency]))
            #             cap.append(max_values[-1][max_index][currency]-1)
            #             x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
            #                     'KRUR': rub_value, 'KUSD': usd_value,
            #                     'BTC': btc_value}
            #             x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
            #                     'KRUR': round(rub_value, 5), 'KUSD': round(usd_value, 5),
            #                     'BTC': round(btc_value,5)}
            #             active_data.append(x)
            # active = pd.DataFrame(active_data)
            date_port = date_port.fillna(method='ffill')

            return fig








def MAX_MIN_bd_conv(currency, func,funcx, dfs, date, portf, fig):
            color = {'KRUR': 'red',  'KUSD':'blue' , 'BTC':'orange', 'white':'white', 'GOLD':'yellow'}
            history = down_bd(portf)
            max_index =0
            port = {}
            cur_port = {}

            # for i in range(len(date)):
            #     date[i] = date[i].strftime('%Y-%m-%d %H:%M:%S')
            
            active_data = []
            max_values = []
            if func == pd.DataFrame.max:
                title = 'График роста капитала, при правильном перераспределении активов' 
            else:
                title = 'График падения капитала, при неправильном перераспределении активов'
            capital = [0]
            cap = [0]
            # fig = make_subplots(rows=1, cols=2, subplot_titles=[title, 'Изменение активов, при их правильном перераспределении'])
            # dfs[0].loc[:,['KRUR', 'KUSD']]
            pre_sum, sum, kief = 0, 0, 0

            for i in range(len(date)-1):
                    const = 0
                    for k in list(history.keys()):
                        if  date[i] in history[k].keys():

                            if k not in cur_port.keys():
                                cur_port[k] = [0]
                            
                            if 'add' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] + history[k][date[i]]['add'])
                                kief = 1
                                # fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                #         y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                #                             mode='lines',
                                #                             line=dict(color=color[k], dash='dash'),
                                #                             name = k,
                                #                             text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(cur_port[k][-1]*dfs[i].loc['KRUR', k],2),
                                #                                                                                                     round(cur_port[k][-1]*dfs[i].loc['KUSD', k],2), 
                                #                                                                                                     round(cur_port[k][-1]*dfs[i].loc['BTC', k],5),

                                #                                                                                                     ),
                                #                             showlegend = False,),
                                #                             row=1,  
                                #                             col=1
                                #                             )
                                

                            if 'withdraw' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] - history[k][date[i]]['withdraw'])
                                # fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                #         y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                #                             mode='lines',
                                #                             line=dict(color=color[k], dash='dash'),
                                #                             name = k,
                                #                             text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(cur_port[k][-1]*dfs[i].loc['KRUR', k],2),
                                #                                                                                                     round(cur_port[k][-1]*dfs[i].loc['KUSD', k],2), 
                                #                                                                                                     round(cur_port[k][-1]*dfs[i].loc['BTC', k],5),

                                #                                                                                                     ),
                                #                             showlegend = False,),
                                #                             row=1,  
                                #                             col=1
                                                            # )
                            
                            const +=1







                    if const == 0 and i>0:
                        pass
                        # for k in list(cur_port.keys()):
                            # fig.add_trace(go.Scatter(x=date[i-1:i+1],
                            #             y= [cur_port[k][-1]* dfs[i-1].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                            #                                 mode='lines',
                            #                                 line=dict(color=color[k]),
                            #                                 name = k,
                            #                                 text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(cur_port[k][-1]*dfs[i].loc['KRUR', k],2),
                            #                                                                                                         round(cur_port[k][-1]*dfs[i].loc['KUSD', k],2), 
                            #                                                                                                         round(cur_port[k][-1]*dfs[i].loc['BTC', k],5),

                            #                                                                                                         ),
                            #                                 showlegend = False,),
                            #                                 row=1,  
                            #                                 col=1
                            #                                 )
                        


                    sum = 0                                        
                    for k in list(cur_port.keys()):
                        list_f = pd.DataFrame(dfs[i-1].loc[:,list(cur_port.keys())])
                        list_s = pd.DataFrame(dfs[i].loc[:,list(cur_port.keys())])
                        max_values.append(1+(list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
                        max = func((list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
                        data1 = list_f.index.name
                        data2 = list_s.index.name
                        max_index = func(max_values[-1])

                        max_index = funcx(max_index)
                        # st.write(max_values[-1], 'max',  max_index)
                        # st.write('max koef',  max_values[-1].loc[currency,max_index])

                        rub_value = max_values[-1][max_index][0]
                        usd_value = max_values[-1][max_index][1]
                        btc_value = max_values[-1][max_index][2]
                        # st.write(rub_value, usd_value,  btc_value)

                        capital.append(capital[-1]* (max_values[-1][max_index][currency]))
                        cap.append(max_values[-1][max_index][currency]-1)
                        x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
                                'KRUR': rub_value, 'KUSD': usd_value,
                                'BTC': btc_value}
                        x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
                                'KRUR': round(rub_value, 5), 'KUSD': round(usd_value, 5),
                                'BTC': round(btc_value,5)}
                        active_data.append(x)

                        active = pd.DataFrame(active_data)
                        cur_port[k][-1] *= max_values[-1].loc[k, max_index]
                        sum += cur_port[k][-1]*dfs[i].loc[currency, k]
                            #тут можжно добавить условие если суммы разные без учетов курса за прошлый и этот промеж то рисуем между этими точкми пунктир в эту дату!или сравнивать длину cur_port
                    # capital.append(sum)


                    if const != 0 and i>0:#ситуация когда были изменения капитала за счет перекладываний!
                        # for k in list(history.keys()):
                        #         if j in list(history[k][date[i]].keys())

                        #     st.write(pre_sum, sum)
                            fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [pre_sum , sum],
                                                            mode='lines',
                                                            line=dict(color='white', dash = 'dash'),
                                                            name = max_index,
                                                            text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=2)

                            fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum , pre_sum ],
                                                        mode='lines',
                                                        line=dict(color=color[max_index]),
                                                        name = max_index,
                                                        text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(pre_sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(pre_sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(pre_sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    elif max_index!=0:

                        fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum , sum  ],
                                                        mode='lines',
                                                        line=dict(color=color[max_index]),
                                                        name = max_index,
                                                        text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    pre_sum = sum
                        
                        

            return fig





def without_action(currency, dfs, date, portf, fig):
            color = {'KRUR': 'red',  'KUSD':'blue' , 'BTC':'orange', 'GOLD':'yellow'}

            history = down_bd(portf)
            port = {}
            cur_port = {}
            # for i in range(len(date)):
            #     date[i] = date[i].strftime('%Y-%m-%d %H:%M:%S')
            active_data = []
            max_values = []
            title = 'График безд' 
            # fig = go.Figure()

            # fig = make_subplots(rows=1, cols=2, subplot_titles=[title, 'Изменение активов, при их правильном перераспределении'])
            # dfs[0].loc[:,['KRUR', 'KUSD']]
            pre_sum, sum = 0, 0 
            for i in range(len(date)-1):
                    const = 0
                    for k in list(history.keys()):
                        if  date[i] in history[k].keys():

                            if k not in cur_port.keys():
                                cur_port[k] = [0]
                            
                            if 'add' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] + history[k][date[i]]['add'])

                                # fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                #         y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                #                             mode='lines',
                                #                             line=dict(color=color[k], dash='dash'),
                                #                             name = k,
                                #                             # text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[0],data.columns[j]],2),
                                #                             #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[1],data.columns[j]],2), 
                                #                             #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[2],data.columns[j]],5),

                                #                             #                                                                         ),
                                #                             showlegend = False,),
                                #                             row=1,  
                                #                             col=1
                                #                             )

                            if 'withdraw' in list(history[k][date[i]].keys()):
                                cur_port[k].append(cur_port[k][-1] - history[k][date[i]]['withdraw'])
                                # fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                #         y= [cur_port[k][-2]* dfs[i].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                                #                             mode='lines',
                                #                             line=dict(color=color[k], dash='dash'),
                                #                             name = k,
                                #                             # text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[0],data.columns[j]],2),
                                #                             #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[1],data.columns[j]],2), 
                                #                             #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[2],data.columns[j]],5),

                                #                             #                                                                         ),
                                #                             showlegend = False,),
                                #                             row=1,  
                                #                             col=1
                                #                             )
                        
                            const +=1
                    if const == 0 and i>0:
                        # for k in list(cur_port.keys()):
                        #     fig.add_trace(go.Scatter(x=date[i-1:i+1],
                        #                 y= [cur_port[k][-1]* dfs[i-1].loc[currency, k], cur_port[k][-1]* dfs[i].loc[currency, k]],
                        #                                     mode='lines',
                        #                                     line=dict(color=color[k]),
                        #                                     name = k,
                        #                                     # text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[0],data.columns[j]],2),
                        #                                     #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[1],data.columns[j]],2), 
                        #                                     #                                                                         round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[2],data.columns[j]],5),

                        #                                     #                                                                         ),
                        #                                     showlegend = False,),
                        #                                     row=1,  
                        #                                     col=1
                        #                                     )
                        pass


                    sum = 0                                        
                    for k in list(cur_port.keys()):
                        sum += cur_port[k][-1]*dfs[i].loc[currency, k]
                            #тут можжно добавить условие если суммы разные без учетов курса за прошлый и этот промеж то рисуем между этими точкми пунктир в эту дату!или сравнивать длину cur_port
                    # capital.append(sum)


                    if const != 0 and i>0:#ситуация когда были изменения капитала за счет перекладываний!
                        # for k in list(history.keys()):
                        #         if j in list(history[k][date[i]].keys())

                        #     st.write(pre_sum, sum)
                            fig.add_trace(go.Scatter(x=[date[i], date[i]],
                                        y= [pre_sum , sum],
                                                            mode='lines',
                                                            line=dict(color='purple', dash = 'dash'),
                                                            name = k,
                                                            text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=2)

                            fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum , pre_sum ],
                                                        mode='lines',
                                                        line=dict(color='purple'),
                                                        name = k,
                                                            text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(pre_sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(pre_sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(pre_sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    else:
                        fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                    y= [pre_sum, sum ],
                                                        mode='lines',
                                                        line=dict(color='purple'),
                                                        name = "Бездействие",
                                                            text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(sum*dfs[i].loc['KRUR', currency],2),
                                                                                                                                    round(sum*dfs[i].loc['KUSD', currency],2), 
                                                                                                                                    round(sum*dfs[i].loc['BTC', currency],5),

                                                                                                                                    ),
                                                        showlegend = False,),
                                                        row=1,  
                                                        col=2
                                                        )
                    pre_sum = sum
            return fig






def MAX_MIN(currency, func,funcx, dfs, date, fig):
            history = down_bd()

    # hystory = {
    #         'KRUR':
    #                 {
    #                      date[5]:{'KUSD':1000},#'BTC':100
    #                      date[2]:{'+': 3000},
                        
    #                      date[6]:{'-': 1000, '+':300},
    #                      date[8]:{'-': 1000},
    # }

            active_data = []
            max_values = []
            if func == pd.DataFrame.max:
                title = 'График роста капитала, при правильном перераспределении активов' 
            else:
                title = 'График падения капитала, при неправильном перераспределении активов'
            capital = [1]
            # c = 0
            cap=[]
            for i in range(1, len(dfs)):
                list_f = pd.DataFrame(dfs[i - 1])
                list_s = pd.DataFrame(dfs[i])
                max_values.append(1+(list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
                max = func((list_s.iloc[:, :] - list_f.iloc[:,:]) / list_f.iloc[:, :])
                data1 = list_f.index.name
                data2 = list_s.index.name
                max_index = func(max_values[i-1])
                max_index = funcx(max_index)
                rub_value = max_values[i-1][max_index][0]
                usd_value = max_values[i-1][max_index][1]
                btc_value = max_values[i-1][max_index][2]
                capital.append(capital[-1]* (max_values[i-1][max_index][currency]))
                cap.append(max_values[i-1][max_index][currency]-1)
                x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
                        'KRUR': rub_value, 'KUSD': usd_value,
                        'BTC': btc_value}
                x = {'Промежуток времени': '{} -- {}'.format(data1, data2), 'Лучший актив': max_index,
                        'KRUR': round(rub_value, 5), 'KUSD': round(usd_value, 5),
                        'BTC': round(btc_value,5)}
                active_data.append(x)
            active = pd.DataFrame(active_data)

            # fig = go.Figure()
            # fig = make_subplots(rows=1, cols=2, subplot_titles=[title, 'Изменение активов, при их правильном перераспределении'])


            data = pd.DataFrame({'KRUR': [st.session_state.portfolio['KRUR']*dfs[0].loc[currency,'KRUR']]*(len(max_values)),
                                'KUSD': [st.session_state.portfolio['KUSD']*dfs[0].loc[currency, 'KUSD']]*(len(max_values)),
                                 'BTC': [st.session_state.portfolio['BTC']*dfs[0].loc[currency, 'BTC']]*(len(max_values))
                                 })
  
            data_portf ={'KRUR': st.session_state.portfolio['KRUR'],
                                 'KUSD': st.session_state.portfolio['KUSD'],
                                 'BTC': st.session_state.portfolio['BTC']
                                 }
            redata = data.iloc[0]

            for i in range(1,len(max_values)):
                for col in dfs[0].columns:
                    data.loc[i,col] = redata[col]*active.loc[i-1,col]
                redata = data.iloc[i]
            data /= data.iloc[0]
            color = {'KRUR': 'red',  'KUSD':'blue' , 'BTC':'orange'}
            color_light = {'KRUR': 'rgba(255, 150, 150, 0.2)',  'KUSD':'rgba(150, 200, 255, 0.2)' , 'BTC':'rgba(255, 200, 150, 0.2)'}
            legend = {'KRUR': True,  'KUSD': True , 'BTC': True}
            for i in range(len(date)-1):
                for j in range(len(data.columns)):
                    if data.columns[j] == active.iloc[i,1]:
                        fig.add_trace(go.Scatter(x=date[i:i+2],
                                                y=capital[i:i+2],
                                                mode='lines',
                                                line=dict(color=color[data.columns[j]]),
                                                name = data.columns[j],
                                                text='''RUB: {:,.2f} <span style="color:green; font-style:italic; font-size:smaller; border: 1px solid #fff; padding: 4px;">+{:,.2f}</span><br>USD:{:,.2f} <span style="color:green; font-style:italic;font-size:smaller">+{:,.4f}</span><br>BTC: {:,.5f}  <span style="color:green; font-style:italic;font-size:smaller">+{:,.5f}'''.format(
                                                                                                                                round(sum*capital[i]*dfs[i].loc[ data.columns[0],currency],2),  round(sum*capital[i]*dfs[i].loc[data.columns[1],currency]  / sum*dfs[0].loc[ data.columns[0],currency],2) ,
                                                                                                                                round(sum*capital[i]*dfs[i].loc[ data.columns[1],currency],2), round(sum*capital[i]*dfs[i].loc[data.columns[1],currency]  / sum*dfs[0].loc[ data.columns[1],currency],5) ,
                                                                                                                                round(sum*capital[i]*dfs[i].loc[ data.columns[2],currency],8), round(sum*capital[i]*dfs[i].loc[data.columns[2],currency]  / sum*dfs[0].loc[ data.columns[2],currency],5) ).replace(',', ' '),
                                                                                                                                
                                                                                                                                showlegend =any(list(legend.values())),
                                                                                                                                ),
                                                                                                                              
                                                row=1,
                                                col=1  
                                                )
                        legend[data.columns[j]] = False
                        


                        fig.add_trace(go.Scatter(x=date[i:i+2],
                                                            y=[i for i in data.iloc[i:i+2,j]],
                                                            mode='lines',
                                                            line=dict(color=color[data.columns[j]]),
                                                            name = data.columns[j],
                                                            text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[0],data.columns[j]],2),
                                                                                                                                  round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[1],data.columns[j]],2), 
                                                                                                                                  round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[2],data.columns[j]],5),

                                                                                                                                  ),
                                                            showlegend = False,),
                                                            row=1,  
                                                            col=2
                                                        
                                                            )

                        continue
                    
                    fig.add_trace(go.Scatter(x=date[i:i+2],
                                                y=[i for i in data.iloc[i:i+2,j]],


                                                mode='lines',
                                                line=dict(color=color[data.columns[j]]),
                                                name = data.columns[j],
                                                text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}'''.format(
                                                                                                                                round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[0],data.columns[j]],2),
                                                                                                                                  round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[1],data.columns[j]],2), 
                                                                                                                                  round(data_portf[data.columns[j]]*data.loc[i,data.columns[j]]*dfs[i].loc[data.columns[2],data.columns[j]],4),
                                                                                                                                  ),
                                                                                                                                  showlegend= False,),
                                                row=1, 
                                                col=2 
                                                )
            # fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=color_light[currency])

            # fig.update_yaxes(showgrid= True, gridwidth=0.5, gridcolor=color_light[currency])
                                                                                                                        
            
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=color_light[currency], row = 1, col=2)

            fig.update_yaxes(showgrid= True, gridwidth=0.5, gridcolor=color_light[currency], row = 1, col=2)

            return fig


def inaction(currency,dfs, date, fig = go.Figure()):
            hystory = down_bd()
            sum = []
            k = 0 
            cur_port = {}
            for k in list(hystory.keys()):
                cur_port[k] = [0]
            all = []
            for i in range(len(date)):
                date[i] = date[i]  + ' 00:00:00'
            start = 0
            sum, pre_sum = 1,1
            for i in range(len(date)-1):
                al = 0

                for k in list(hystory.keys()):
                    if  date[i] in hystory[k].keys():
                        cur_port[k].append(cur_port[k][-1] + hystory[k][date[i]]['add']*dfs[i].loc[option, k])
                        al += cur_port[k][-1] + hystory[k][date[i]]['add']*dfs[i].loc[option, k]
                all.append(al)


            for j in range(len(date)):              
                if j != 0 :
                    k += 1
                    fig.add_trace(go.Scatter(x=date[j-1:j+1],
                                                                y=all[j-1:j+1],

                                                                mode='lines',
                                                                line=dict(color='purple'),
                                                                # name = col,
                                                                # text='''в рублях:{},<br> в долларах:{},<br> в биткойнах: {}, {}'''.format(sumall *dfs[j].loc[col, "KRUR"],
                                                                #                                                                     sumall *dfs[j].loc[col, "KUSD"], 
                                                                #                                                                     sumall *dfs[j].loc[col, "BTC"],
                                                                                                                                  
                                                                                                                        
                                                                                                                                ))

                                                            
                         
            # return fig
            return st.plotly_chart(fig)


def experience(dfs, date, currency, fig= make_subplots(rows=1, cols=2, subplot_titles = [ 'История изменения активов', 'Изменение капитала'])):

    color_light = {'KRUR': 'rgba(255, 150, 150, 0.2)',  'KUSD':'rgba(150, 200, 255, 0.2)' , 'BTC':'rgba(255, 200, 150, 0.2)'}

    # portfel = {
    #     'KRUR': 10000,
    #     'KUSD': 100,
    #     'BTC' : 0.02
    # }
    # hystory = {
    #         'KRUR':
    #                 {
    #                      date[5]:{'KUSD':1000},#'BTC':100
    #                      date[2]:{'+': 3000},
                        
    #                      date[6]:{'-': 1000, '+':300},
    #                      date[8]:{'-': 1000},


    #                 },

    #         'KUSD':
    #                 {
    #                      date[3]:{'+': 120},
    #                      date[4]:{'-': 50},

    #                      date[9]:{'+': 30},
    #                      date[13]:{'+': 30},
    #                      date[15]:{'+': 30},


    #                 },
    #         'BTC':
    #                 {
 
    #                     date[15]:{'+':0.04}                     
    #                 },

    # }
    hystory = down_bd()
    port = {

        # 'KRUR': 10000,
        # 'KUSD': 100,
        # 'BTC' : 0.02
    }   


    color = {'KRUR': 'red',  'KUSD':'blue' , 'BTC':'orange'}
    cur_port = {
        # 'KRUR': [10000],
        # 'KUSD': [100],
        # 'BTC' : [0.02]
    }
    # fig = make_subplots(rows=1, cols=2, subplot_titles = [ 'История изменения активов', 'Изменение капитала'])

    for i in range(len(date)):
        date[i] = date[i].strftime('%Y-%m-%d %H:%M:%S')
    start = 0
    sum, pre_sum = 1,1

    for i in range(len(date)-1):
        
        for k in list(hystory.keys()):


            # time = datetime.strptime(list(hystory[k].keys()), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            # time = time.strftime("%Y-%m-%d")
            

            # st.write(hystory[k].keys())
            # st.write(date[i])

            if k not in port.keys() and date[i] in hystory[k].keys():
                cur_port[k] = []
                cur_port[k].append(hystory[k][date[i]]['add'])#проблема
                # st.write(cur_port, ' cur_port')

                port[k] =  0 #hystory[k][date[i]]['add']
                # st.write(cur_port[k][0]['add'])

                start += cur_port[k][0] * dfs[0].loc[option, k]


        # if date[i] in hystory.keys():
        #     if 'add' in hystory[date[i]].keys():
        #           port[hystory[date[i]]['add']]
        #     if 'withdraw' in hystory[date[i]].keys():
        #         del  port[hystory[date[i]]['withdraw']]
        for j in list(cur_port.keys()):
            
            try:

                if  list(hystory[j][date[i]].keys())[0] == 'add':

                    # fig.add_trace(go.Scatter(x=[date[i],date[i]],
                                            
                    #                      y = [port[j] * dfs[i].loc[option, j], (port[j]* dfs[i].loc[option, j] + hystory[j][date[i]]['add']* dfs[i].loc[option, j])],
                                                            
                    #                                         mode='lines',
                    #                                         line=dict(color=color[j], dash='dash'),
                    #                                         name = j,
                    #                                         text=''' + {} '''.format(hystory[j][date[i]]['add']
                                                                                                                    
                    #                                                                                                               )
                    #                                         ),
                    #                                         row=1,
                    #                                         col=1) 
                    port[j] =  (port[j] + hystory[j][date[i]]['add'])
                    cur_port[j].append(cur_port[j][-1] + hystory[j][date[i]]['add'])
                elif list(hystory[j][date[i]].keys())[0] == 'withdraw':
                    # fig.add_trace(go.Scatter(x=[date[i],date[i]],
                                            
                    #                      y = [port[j]* dfs[i].loc[option, j], (port[j]* dfs[i].loc[option, j] - hystory[j][date[i]]['withdraw']* dfs[i].loc[option, j])],
                                             
                                                            
                    #                                         mode='lines',
                    #                                         line=dict(color=color[j], dash='dash'),
                    #                                         name = j,
                    #                                         text=''' - {} '''.format(hystory[j][date[i]]['withdraw']
                                                                                                                    
                    #                                                                                                               )
                    #                                         ),
                    #                                         row=1,
                    #                                         col=1)
                    port[j] = (port[j] - hystory[j][date[i]]['withdraw'])
                    cur_port[j].append(cur_port[j][-1] - hystory[j][date[i]]['withdraw'])
             
                for k in cur_port.keys():
                    
                    if date[i] in hystory[j].keys()  and k in list(hystory[j][date[i]].keys()):
                        
                        # fig.add_trace(go.Scatter(x=[date[i],date[i]],
                    
                        #             y = [port[j] * dfs[i].loc[option, j], (port[j]* dfs[i].loc[option, j] - hystory[j][date[i]][k]* dfs[i].loc[option, j])],

                        #             mode='lines',
                        #             line=dict(color= color[k], dash='dash'),
                        #             name = j,
                        #             text=''' - {} '''.format(hystory[j][date[i]][k]
                                                                                            
                        #                                                                                     )
                        #             ),
                        #             row=1,
                        #             col=1)

                        port[j] = (port[j] - hystory[j][date[i]][k])
                        # cur_port[j].append(cur_port[j][-1] - hystory[j][date[i]][k])
                        cur_port[j][-1] -= hystory[j][date[i]][k]

                        # fig.add_trace(go.Scatter(x=[date[i],date[i]],
                    
                        #             y = [port[k] * dfs[i].loc[option, k], (port[k]* dfs[i].loc[option, k] + hystory[j][date[i]][k] * dfs[i].loc[option, j])],

                                    
                        #             mode='lines',
                        #             line=dict(color=color[j] , dash='dash'),
                        #             name = j,
                        #             text=''' + {} '''.format(hystory[j][date[i]][k]
                                                                                            
                        #                                                                                     )
                        #             ),
                        #             row=1,
                        #             col=1) 
                        port[k] = (port[k] + hystory[j][date[i]][k] * dfs[i].loc[k, j])
                        # cur_port[k].append(cur_port[k][-1] + hystory[j][date[i]][k]* dfs[i].loc[k, j])
                        # fig.add_trace(go.Scatter(x=[date[i],date[i+1]],
                                            
                        #                  y = [port[k] * dfs[i].loc[option, k], port[k]* dfs[i+1].loc[option, k]],
                                                            
                        #                                     mode='lines',
                        #                                     line=dict(color= color[k]),
                        #                                     name = j,

                        #                                     text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                        #                                     cur_port[k][-1] * dfs[i].loc['KRUR',k],
                        #                                     cur_port[k][-1] * dfs[i].loc['KUSD',k],
                        #                                     cur_port[k][-1] * dfs[i].loc['BTC', k])
                                                                                                                    
                        #                                                                                                           )
                        #                                     ,
                        #                                     row=1,
                        #                                     col=1 )
                cur_port[j].append(cur_port[j][-1])
                if i!=len(date)-1:
                    cur_port[j].append(cur_port[j][-1])

                    # fig.add_trace(go.Scatter(x=[date[i],date[i+1]],
                                            
                    #                      y = [port[j]* dfs[i].loc[option, j], port[j]* dfs[i+1].loc[option, j]],

                    #                                         mode='lines',
                    #                                         line=dict(color=color[j]),
                    #                                         name = j,

                    #                                         text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                    #                                         cur_port[j][-1] * dfs[i].loc['KRUR',j],
                    #                                         cur_port[j][-1] * dfs[i].loc['KUSD',j],
                    #                                         cur_port[j][-1] * dfs[i].loc['BTC', j])
                                                                                                                    
                    #                                                                                                               )
                    #                                         ,
                    #                                         row=1,
                    #                                         col=1 )
            except:

                try:
                     if i!=len(date)-1 and   "KUSD" not in list(hystory[j][date[i]].keys()):
                        pass
                        # fig.add_trace(go.Scatter(x=[date[i],date[i+1]],
                                                        
                        #                             y = [port[j] * dfs[i].loc[option, j], port[j]* dfs[i+1].loc[option, j]],
                                                                        
                        #                                                 mode='lines',
                        #                                                 line=dict(color= color[j]),
                        #                                                 name = j,

                        #                                                 text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                        #                                                 cur_port[j][-1] * dfs[i].loc['KRUR',j],
                        #                                                 cur_port[j][-1] * dfs[i].loc['KUSD',j],
                        #                                                 cur_port[j][-1] * dfs[i].loc['BTC', j])
                                                                                                                                
                        #                                                                                                                     )
                        #                                                 ,
                        #                                                 row=1,
                        #                                                 col=1 )
                except:
                    if i!=len(date)-1 : # and date[i] in list(hystory[j].keys())   and i not in list(hystory[j][date[i]].keys()):
                        
                        # fig.add_trace(go.Scatter(x=[date[i],date[i+1]],
                                                        
                        #                             y = [port[j] * dfs[i].loc[option, j], port[j]* dfs[i+1].loc[option, j]],
                                                                        
                        #                                                 mode='lines',
                        #                                                 line=dict(color= color[j]),
                        #                                                 name = j,

                        #                                                 text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                        #                                                 cur_port[j][-1] * dfs[i].loc['KRUR',j],
                        #                                                 cur_port[j][-1] * dfs[i].loc['KUSD',j],
                        #                                                 cur_port[j][-1] * dfs[i].loc['BTC', j])
                                                                                                                                
                        #                                                                                                                     )
                        #                                                 ,
                        #                                                 row=1,
                        #                                                 col=1 )
                        cur_port[j].append(cur_port[j][-1])

                    
        sum_change = 0
        for k in list(cur_port.keys()):
            sum += cur_port[k][-1] * dfs[i].loc[option, k]
            sum_change += (cur_port[k][-1] - cur_port[k][-2]) * dfs[i].loc[option, k]

        c = 0
        for j in list(port.keys()):
                try:
                    if list(hystory[j][date[i]].keys())[0] == 'add' or list(hystory[j][date[i]].keys())[0] == 'withdraw':
                        c += 1
                except:
                    pass
        if c>0:
            fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                                    
                                                                    y = [pre_sum, pre_sum* dfs[i].loc[option, k]/ dfs[i-1].loc[option, k]],
                                                                    mode='lines',
                                                                    line=dict(color='white'),
                                                                    name = 'Капитал',
                                                                    text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                                                                        sum * dfs[i-1].loc['KRUR',option],
                                                                        sum * dfs[i-1].loc['KUSD',option],
                                                                        sum * dfs[i-1].loc['BTC', option]                  
                                                                                                                                            )
                                                                    ),
                                                            # row=1,
                                                            # col=2 
                                                            )
            fig.add_trace(go.Scatter(x=[date[i],date[i]],
                                                y = [pre_sum* dfs[i].loc[option, k]/ dfs[i-1].loc[option, k], sum],
                                                mode='lines',
                                                line=dict(color='white', dash = 'dash'),
                                                name = 'Капитал',
                                                        text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                                                            sum * dfs[i].loc['KRUR',option],
                                                            sum * dfs[i].loc['KUSD',option],
                                                            sum * dfs[i].loc['BTC', option]
                                                                                                    
                                                                                                                        )
                                                ),
                                                            # row=1,
                                                            # col=2 
                                                            )


                    
        else:
                fig.add_trace(go.Scatter(x=date[i-1:i+1],
                                        
                                                        y = [pre_sum, sum],
                                                        mode='lines',
                                                        line=dict(color='white'),
                                                        name = 'Капитал',
                                                        text='''RUB:{},<br>USD:{},<br>BTC:{}'''.format(
                                                            sum * dfs[i-1].loc['KRUR',option],
                                                            sum * dfs[i-1].loc['KUSD',option],
                                                            sum * dfs[i-1].loc['BTC', option]

                                                                                                            
                                                                                                                                )
                                                        ),
                                                            # row=1,
                                                            # col=2 
                                                            )
                                                            
        pre_sum = sum
        sum = 0

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=color_light[currency])

    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=color_light[currency])

    # Отключаем функцию приближения (zoom)
    # fig.update_layout(dragmode='select', showlegend=False, xaxis=dict(constrain='domain'), yaxis=dict(constrain='domain'))
    return st.plotly_chart(fig)

def uno(dfs, date, currency, fig):


    coin = 10000
    # fig = go.Figure()
    for i in range(len(dfs)-1):
                for col in dfs[i].columns:
                        if col == 'KRUR':
                            color = 'red'
                        elif col == 'KUSD':
                            color = 'blue'
                        else:
                            color = 'orange'
                        fig.add_trace(go.Scatter(x=date[i:i+2],
                                                            y=[coin*dfs[i].loc[col,option]/dfs[0].loc[col,option],coin*dfs[i+1].loc[col, option]/dfs[0].loc[col,option] ],

                                                            mode='lines',
                                                            line=dict(color=color),
                                                            name = col,
                                                            )) 
    return st.plotly_chart(fig)

    
     
# def exp():
    
#     # Список элементов для выпадающего списка
#     options = ['Max', 'Min', 'inaction', 'Users']

#     # Виджет выпадающего списка
#     selected_option = st.multiselect("Выберите вариант:", options)
#     # st.write(selected_option)
#     fig = go.Figure()
#     if 'Max' in selected_option:
#         MAX_MIN(option, pd.DataFrame.max, pd.Series.idxmax, dfs,date, sum,st.session_state.portfolio, fig)
#     if 'Min' in selected_option:
#         MAX_MIN(option, pd.DataFrame.min, pd.Series.idxmin, dfs,date, sum, st.session_state.portfolio, fig)
#     if 'inaction' in selected_option:
#         inaction(option, dfs, date, fig, sum)
#     if 'Users' in selected_option:
#          experience(dfs, date, option, fig)
#     # fig.update_layout(yaxis_type="log")
#     # st.write(sum)
#     return fig




def exp(option, dfs,date, currency):
    # Список элементов для выпадающего списка
    color_light = {'KRUR': 'rgba(255, 150, 150, 0.2)',  'KUSD':'rgba(150, 200, 255, 0.2)' , 'BTC':'rgba(255, 200, 150, 0.2)', 'GOLD': 'rgba(255, 255, 150, 0.2)'}


    options = ['Max', 'Min', 'inaction', 'Users']
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Активы', 'Изменение капитала'])

    # Виджет выпадающего списка
    selected_option = st.multiselect("Выберите вариант:", options)
    # st.write(selected_option)
    # fig = go.Figure()
    if 'Max' in selected_option:
        MAX_MIN_bd_conv(option, pd.DataFrame.max, pd.Series.idxmax, dfs,date, portf, fig)
    if 'Min' in selected_option:
         MAX_MIN_bd_conv(option, pd.DataFrame.min, pd.Series.idxmin, dfs,date, portf, fig)
    if 'inaction' in selected_option:
        without_action(option, dfs, date, portf, fig)
    if 'Users' in selected_option:
         MAX_MIN_bd(option, dfs,date, portf, fig)
        #  st.write(date_port)

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=color_light[currency])

    fig.update_yaxes(showgrid= True, gridwidth=0.5, gridcolor=color_light[currency])
    return st.plotly_chart(fig)
# dfs,date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram  = download(start_date, end_date, frequency)

dfs, date, data_rub_USD, data_btc_usd, gold_data, gold_usd_gram = download(start_date, end_date, frequency, 'data.pkl', update=False)

# st.write('data_rub_USD',  data_rub_USD, len(data_rub_USD))
# st.write('data_btc_usd', data_btc_usd, len(data_btc_usd))
# st.write('gold_usd_gram', gold_usd_gram, len(gold_usd_gram))
        





for i in range(len(date)):
    date[i] = date[i].strftime('%Y-%m-%d %H:%M:%S')
# st.write(date)









exp(option, dfs,date, option)

##############################################################################################################################
volatilit = pd.DataFrame(columns=['KRUR', 'KUSD', 'BTC', 'Gold'])#.set_index('Date')




def volatility(volatilit, start, end):
    # Преобразуйте строки в объекты даты
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')    
    ind = '{}:{}'.format(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    volatilit.loc[ind, 'KRUR'] = np.std(1/data_rub_USD[start:end])
    volatilit.loc[ind, 'BTC'] = np.std(data_btc_usd[start:end])
    volatilit.loc[ind, 'Gold'] = np.std(gold_usd_gram[start:end])
    volatilit.loc[ind, 'KUSD'] = 1/volatilit.loc[ind, 'KRUR']
    return volatilit




def volatility_sred(volatilit, start, end):
 
    ind = '{}:{}'.format(pd.to_datetime(start).strftime('%Y-%m-%d'), pd.to_datetime(end).strftime('%Y-%m-%d'))  # Изменяем формат для индекса

    volatilit.loc[ind, 'KRUR'] = np.std(1/(data_rub_USD[start:end]/ np.mean(data_rub_USD[start:end])))*100
    volatilit.loc[ind, 'BTC'] = np.std(data_btc_usd[start:end]/np.mean(data_btc_usd[start:end]))*100
    volatilit.loc[ind, 'Gold'] = np.std(gold_usd_gram[start:end]/np.mean(gold_usd_gram[start:end]))*100
    volatilit.loc[ind, 'KUSD'] = np.std(data_rub_USD[start:end]/ np.mean(data_rub_USD[start:end])) * 100

    return volatilit

def vol_proc(volatilit, start, end):
     
    ind = '{}:{}'.format(pd.to_datetime(start).strftime('%Y-%m-%d'), pd.to_datetime(end).strftime('%Y-%m-%d'))  # Изменяем формат для индекса
    volatilit.loc[ind, 'KRUR'] = np.std(1/(data_rub_USD[start:end]/data_rub_USD[start]))*100
    volatilit.loc[ind, 'BTC'] = np.std(data_btc_usd[start:end]/data_btc_usd[start])*100
    volatilit.loc[ind, 'Gold'] = np.std(gold_usd_gram[start:end]/data_rub_USD[start])*100
    volatilit.loc[ind, 'KUSD'] = 0#np.std((data_rub_USD[start:end]/data_rub_USD[start]))*100

    return volatilit

def prof(x_start_str, x_end_str,  dfs, option):
    x_start = datetime.datetime.strptime(x_start_str, '%Y-%m-%d %H:%M:%S')
    x_end = datetime.datetime.strptime(x_end_str, '%Y-%m-%d %H:%M:%S')
    profitability_abs= 0
    global date_port
    profic =  pd.DataFrame(columns=['KRUR', 'KUSD', 'BTC', 'GOLD'])
    beginning = dfs[0].index.name
    start = x_start-beginning
    start = start.days
    end =  start + (x_end - x_start).days + 1
    a = 0
    for j in range(start+1, end):
        change = dfs[j]/dfs[j-1] - 1 
        for i in date_port.columns:
            profitability_abs += (date_port.loc[(x_start + datetime.timedelta(days=a)).strftime('%Y-%m-%d %H:%M:%S'), i]*dfs[j-1].loc[option,i] * change.loc[option, i])
        # st.write(profitability_abs,x_start + datetime.timedelta(days=a))

        a += 1
    st.write(profitability_abs)


def risk(x_start_str, x_end_str, dfs, option, corr,std ):
    global date_port

    x_start = datetime.datetime.strptime(x_start_str, '%Y-%m-%d %H:%M:%S')
    x_end = datetime.datetime.strptime(x_end_str, '%Y-%m-%d %H:%M:%S')
    beginning = dfs[0].index.name
    start = x_start-beginning
    start = start.days
    end =   (x_end - beginning).days
    sum = 0
    for i in ['KRUR', 'KUSD', 'BTC', 'GOLD']:
         sum += date_port.loc[x_end_str, i]*dfs[end].loc[option, i]
    el_1= 0
    el_2 = 0
    w_active = {'KRUR':0, 'KUSD':0, 'BTC':0, 'GOLD':0}
    active = 0
    for i in ['KRUR', 'KUSD', 'BTC', 'GOLD']:
        w_active[i] =  date_port.loc[x_end_str, i]*dfs[end].loc[option, i]/sum
        el_1 += w_active[i]**2 * (std.iloc[0, active] )**2
        active +=1
    active1 = 0
    active2= 0

    for i in ['KRUR', 'KUSD', 'BTC', 'GOLD']:
        active2 = 0

        for j in ['KRUR', 'KUSD', 'BTC', 'GOLD']:
             el_2 += w_active[i]*w_active[j]*corr.iloc[active1,active2]*(std.iloc[0, active1]*std.iloc[0, active2])
             active2 +=1
        active1 +=1

    total = (el_1+2*el_2)**0.5
    st.write('Риск',total)

# # Количество дней, которое мы хотим добавить
# days_to_add = 7

# # Создаем объект timedelta для добавления к дате
# delta = timedelta(days=days_to_add)

# # Добавляем количество дней к дате
# new_date = date + delta


#доходность это проценты с (i,i+1) умноженные на бабки которые были в i
    



index = ['RUB','USD', 'BTC', 'Gold']
columns = ['RUB', 'USD', 'BTC', 'Gold']
corr = pd.DataFrame(index=index, columns=columns)
def func_corr(corr, start, end, lst):
    lst = dict(zip( ['RUB', 'USD', 'BTC', 'Gold'], lst))
    for i in list(corr.columns):
         for j in list(corr.columns):
            corr.loc[i, j] = lst[i].corr(lst[j])                 
    # st.write(corr)
    return corr


dat = date.copy()
# dat = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dat]
dat = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dat]

# Получаем значения начала и конца диапазона при наведении курсора на график
x_start, x_end = st.slider("Выберите временной диапазон", 
                            min_value=dat[0], 
                           max_value=dat[-1], 
                           value=(dat[0], dat[-1]))

# Конвертируем обратно в строки
x_start_str = x_start.strftime('%Y-%m-%d %H:%M:%S')
x_end_str = x_end.strftime('%Y-%m-%d %H:%M:%S')

st.write(f"Волатильность в абсолютных знач")
# st.write(data_rub_USD[x_start_str:x_end_str])
st.write(volatility(volatilit, x_start_str, x_end_str))
st.write(f"Волатильность в процентом знач")

# st.write(volatility_sred(volatilit, x_start_str, x_end_str))

std = vol_proc(volatilit, x_start_str, x_end_str)
st.write(std)

st.write(f"корреляция активов в процентных изменениях")

corr = func_corr(corr,x_start_str, x_end_str, [1/data_rub_USD[x_start_str:x_end_str]/data_rub_USD[x_start_str],data_rub_USD[x_start_str:x_end_str]/data_rub_USD[x_start_str], data_btc_usd[x_start_str:x_end_str]/data_btc_usd[x_start_str], gold_usd_gram[x_start_str:x_end_str]/gold_usd_gram[x_start_str]])
st.write(corr)
st.write(f"доходность")

# Предположим, что dfs[0].index.name и dfs[5].index.name - это объекты Timestamp

prof(x_start_str, x_end_str, dfs, option)
# st.write(dfs[5].index.name, (dfs[0].index.name, dfs[5].index.name - dfs[0].index.name).days)

risk(x_start_str, x_end_str, dfs, option, corr,std )




# st.write([1/data_rub_USD['2023-01-03':'2023-03-01'],data_rub_USD['2023-01-03':'2023-03-01'], data_btc_usd['2023-01-03':'2023-03-01'], gold_usd_gram['2023-01-03':'2023-03-01']])

##############################################################################################################################




# волатильность
# чистый доход
# риск
#коэф корреляции
