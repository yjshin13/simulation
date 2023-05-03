import backtest
import streamlit as st
from datetime import datetime
import backtest_graph2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
st.set_page_config(layout="wide")
file = st.file_uploader("Upload investment universe & price data", type=['xlsx', 'xls', 'csv'])
st.warning('Upload data.')

if file is not None:

    @st.cache
    def load_data(file_path):
        df = pd.read_excel(file_path, sheet_name="data",
                           names=None, dtype={'Date': datetime}, index_col=0, header=2)

        df2 = pd.read_excel(file_path, sheet_name="data",
                           names=None, index_col=0, header=0, nrows=1)

        return df, df2


    price, weight = load_data(file)

    price_list = list(map(str, price.columns))
    select = st.multiselect('Input Assets', price_list, price_list)
    input_list = price.columns[price.columns.isin(select)]
    input_price = price[input_list]

    @st.cache
    def summit(x):
        return x

    summit = summit(1)


    if (st.button('Summit') or ('input_list' in st.session_state)):

        with st.expander('Portfolio', expanded=True):

            input_price = input_price.dropna()

            col40, col41, col42, col43, col44, col45, col46, col47 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

            with col40:

                start_date = st.date_input("Start", value=input_price.index[0])
                start_date = datetime.combine(start_date, datetime.min.time())

            with col41:

                end_date = st.date_input("End", value=input_price.index[-1])
                end_date = datetime.combine(end_date, datetime.min.time())

            with col42:

                option1 = st.selectbox(
                    'Data Frequency', ('Daily', 'Monthly'))

            with col43:

                rebal = st.selectbox('Rebalancing', ( 'Monthly', 'Daily', 'Quarterly', 'Yearly'))

            with col44:

                commission = st.number_input('Commission(%)')

            if option1 == 'Daily':
                daily = True
                monthly = False
                annualization = 365
                freq = 1

            if option1 == 'Monthly':
                daily = False
                monthly = True
                annualization = 12
                freq = 2
            #
            st.session_state.input_list = input_list

            if daily == True:
                st.session_state.input_price = input_price[
                    (input_price.index >= start_date) & (input_price.index <= end_date)]

            if monthly == True:
                st.session_state.input_price = input_price[(input_price.index >= start_date)
                                                           & (input_price.index <= end_date)
                                                           & (input_price.index.is_month_end == True)].dropna()


            st.session_state.input_price = pd.concat([st.session_state.input_price,
                                                      pd.DataFrame({'Cash': [1]*len(st.session_state.input_price)},
                                                      index=st.session_state.input_price.index)], axis=1)

            col1, col2, col3 = st.columns([1, 1, 1])

            slider = pd.Series()
            #
            # st.write(input_price.columns)

            st.write("Allocation(%)")

            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            for i, k in enumerate(st.session_state.input_list, start=0):

                if i % 4 == 0:
                    with col1:
                        slider[k] = st.number_input(str(k), float(0), float(100),  float(weight[k]*100), 0.5)

                if i % 4 == 1:
                    with col2:
                        slider[k] = st.number_input(str(k), float(0), float(100),  float(weight[k]*100), 0.5)

                if i % 4 == 2:
                    with col3:
                        slider[k] = st.number_input(str(k), float(0), float(100),  float(weight[k]*100), 0.5)

                if i % 4 == 3:
                    with col4:
                        slider[k] = st.number_input(str(k), float(0), float(100),  float(weight[k]*100), 0.5)




            slider['Cash'] = 100 - slider.sum()
            st.write(str("Total Weight:   ") + str((slider.sum()-slider['Cash']).round(2)) + str("%"))

            #########################[Graph Insert]#####################################

            if st.button('Simulation'):


                st.session_state.slider = (slider*0.01).tolist()
                st.session_state.portfolio_port, st.session_state.allocation_f,\
                    st.session_state.allocation= backtest.simulation(st.session_state.input_price,st.session_state.slider,
                                                                                                   commission,
                                                                                                   rebal)

                st.session_state.alloc =  st.session_state.allocation_f.copy()
                #st.session_state.alloc[st.session_state.alloc.index.is_month_end==True] = st.session_state.allocation_f.iloc[0]
                st.session_state.ret = (st.session_state.input_price.iloc[1:] / st.session_state.input_price.shift(1).dropna()-1)

                st.session_state.contribution = (st.session_state.ret* (st.session_state.alloc.shift(1).dropna())).dropna().sum(axis=0)

                if monthly == True:
                    st.session_state.portfolio_port = st.session_state.portfolio_port[st.session_state.portfolio_port.index.is_month_end==True]


                st.session_state.drawdown = backtest.drawdown(st.session_state.portfolio_port)
                st.session_state.input_price_N = st.session_state.input_price[(st.session_state.input_price.index>=st.session_state.portfolio_port.index[0]) &
                                                                            (st.session_state.input_price.index<=st.session_state.portfolio_port.index[-1])]
                st.session_state.input_price_N = 100 * st.session_state.input_price_N / st.session_state.input_price_N.iloc[0, :]


                st.session_state.portfolio_port.index = st.session_state.portfolio_port.index.date
                st.session_state.drawdown.index = st.session_state.drawdown.index.date
                st.session_state.input_price_N.index = st.session_state.input_price_N.index.date
                st.session_state.alloc.index = st.session_state.alloc.index.date



                st.session_state.result = pd.concat([st.session_state.portfolio_port,
                                                     st.session_state.drawdown,
                                                     st.session_state.input_price_N,
                                                     st.session_state.alloc],
                                                    axis=1)
                # st.session_state.result = st.session_state.result[(st.session_state.result.index>=st.session_state.portfolio_port.index[0]) &
                #                                                   (st.session_state.result.index<=st.session_state.portfolio_port.index[-1])]




            if 'slider' in st.session_state:

                START_DATE = st.session_state.portfolio_port.index[0].strftime("%Y-%m-%d")
                END_DATE = st.session_state.portfolio_port.index[-1].strftime("%Y-%m-%d")
                Anuuual_RET = round(float(((st.session_state.portfolio_port[-1] / 100) ** (annualization / (len(st.session_state.portfolio_port) - 1)) - 1) * 100), 2)
                Anuuual_Vol = round(float(np.std(st.session_state.portfolio_port.pct_change().dropna())*np.sqrt(annualization)*100),2)
                Anuuual_Sharpe = round(Anuuual_RET/Anuuual_Vol,2)
                MDD  =round(float(min(st.session_state.drawdown) * 100), 2)
                Daily_RET = st.session_state.portfolio_port.pct_change().dropna()
                
                st.write(" ")

                col50, col51, col52, col53, col54 = st.columns([1, 1, 1, 1, 1])


                with col50:
                    st.info("Period: " + str(START_DATE) + " ~ " + str(END_DATE))

                with col51:
                    st.info("Annual Return: "+str(Anuuual_RET)+"%")

                with col52:
                    st.info("Annual Volatility: " + str(Anuuual_Vol) +"%")

                with col53:

                    st.info("Sharpe Ratio: " + str(Anuuual_Sharpe))

                with col54:

                    st.info("MDD: " + str(MDD) + "%")


                col21, col22, col23, col24 = st.columns([0.8, 0.8, 3.5, 3.5])

                with col21:
                    st.write('NAV')
                    st.dataframe(st.session_state.portfolio_port.round(2))

                    st.download_button(
                        label="Download",
                        data=st.session_state.result.to_csv(index=True),
                        mime='text/csv',
                        file_name='Result.csv')

                with col22:
                    st.write('MDD')
                    st.dataframe(st.session_state.drawdown.apply(lambda x: '{:.2%}'.format(x)))

                    # st.download_button(
                    #     label="MDD",
                    #     data=st.session_state.drawdown.apply(lambda x: '{:.2%}'.format(x)).to_csv(index=True),
                    #     mime='text/csv',
                    #     file_name='MAX Drawdown.csv')

                with col23:
                    st.write('Normalized Price')
                    st.dataframe((st.session_state.input_price_N).
                                  astype('float64').round(2))
                    #
                    # st.download_button(
                    #     label="Assets",
                    #     data=(100*st.session_state.input_price/st.session_state.input_price.iloc[0,:]).
                    #               astype('float64').round(2).to_csv(index=True),
                    #     mime='text/csv',
                    #     file_name='Assets.csv')

                with col24:
                    st.write('Floating Weight')
                    st.dataframe(st.session_state.alloc.applymap('{:.2%}'.format))
                    #
                    # st.download_button(
                    #     label="Allocation",
                    #     data=st.session_state.alloc.applymap('{:.2%}'.format).to_csv(index=True),
                    #     mime='text/csv',
                    #     file_name='Allocation.csv')

                st.write(" ")


                col31, col32 = st.columns([1, 1])

                with col31:
                    st.write("Net Asset Value")
                    st.pyplot(backtest_graph2.line_chart(st.session_state.portfolio_port, ""))

                with col32:
                    st.write("MAX Drawdown")
                    st.pyplot(backtest_graph2.line_chart(st.session_state.drawdown, ""))


                col61, col62 = st.columns([1, 1])

                with col61:

                    st.download_button(
                        label="Download",
                        data=st.session_state.portfolio_port.to_csv(index=True),
                        mime='text/csv',
                        file_name='Daily Contribution.csv')

                with col62:

                    st.download_button(
                        label="Download",
                        data=st.session_state.drawdown.to_csv(index=True),
                        mime='text/csv',
                        file_name='Correlation.csv')



                st.write(" ")



                col_a, col_b, = st.columns([1,1])


                with col_a:

                    st.write("Performance Contribution")
                    st.session_state.contribution.index = pd.Index(st.session_state.contribution.index.map(lambda x: str(x)[:7]))



                    x = (st.session_state.contribution * 100)
                    y = st.session_state.contribution.index

                    fig_bar, ax_bar = plt.subplots(figsize=(18, 11))
                    width = 0.75  # the width of the bars
                    bar = ax_bar.barh(y, x, color="lightblue", height=0.8, )

                    for bars in bar:
                        width = bars.get_width()
                        posx = width + 0.01
                        posy = bars.get_y() + bars.get_height() * 0.5
                        ax_bar.text(posx, posy, '%.1f' % width, rotation=0, ha='left', va='center', fontsize=13)

                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.xlabel('Contribution(%)', fontsize=15, labelpad=20)
                    #ax_bar.margins(x=0, y=0)

                    st.pyplot(fig_bar)




                with col_b:
                    st.write("Correlation Matrix")

                    # Increase the size of the heatmap.
                    fig2 = plt.figure(figsize=(15, 8.3))
                    # plt.rc('font', family='Malgun Gothic')
                    plt.rcParams['axes.unicode_minus'] = False

                    st.session_state.corr = st.session_state.input_price.drop(['Cash'], axis=1).pct_change().dropna().corr().round(2)
                    st.session_state.corr.index = pd.Index(st.session_state.corr.index.map(lambda x: str(x)[:7]))
                    st.session_state.corr.columns = st.session_state.corr.index
                    # st.session_state.corr.columns = pd.MultiIndex.from_tuples([tuple(map(lambda x: str(x)[:7], col)) for col in st.session_state.corr.columns])

                    heatmap = sns.heatmap(st.session_state.corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')

                    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=12)

                    st.pyplot(fig2)

                col71, col72 = st.columns([1, 1])

                with col71:

                    st.download_button(
                        label="Download",
                        data=(st.session_state.ret* (st.session_state.alloc.shift(1).dropna())).dropna().to_csv(index=True),
                        mime='text/csv',
                        file_name='Contribution.csv')

                with col72:

                    st.download_button(
                        label="Download",
                        data=st.session_state.corr.to_csv(index=True),
                        mime='text/csv',
                        file_name='Correlation.csv')




