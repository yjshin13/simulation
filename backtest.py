import pandas as pd


def cleansing(assets_data=pd.DataFrame(), alloc=list()):

    alloc = pd.DataFrame(alloc).T

    assets_data = pd.DataFrame(assets_data,
                           index=pd.date_range(start=assets_data.index[0],
                                                end=assets_data.index[-1], freq='D')).fillna(method='ffill')

    allocation = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    allocation[:] = alloc
    allocation = allocation[allocation.index.is_month_end == True]

    return assets_data, allocation


def simulation(assets_data, allocation, date='1900-01-01'):

    assets_data = assets_data[assets_data.index>=date]

    if type(allocation)==list:
        assets_data ,allocation = cleansing(assets_data, allocation)

    portfolio = pd.DataFrame(index=assets_data.index, columns=['nav']).squeeze()
    portfolio = portfolio[portfolio.index >= allocation.index[0]]
    portfolio[0] = 100

    k = 0
    j_rebal = 0
    i_rebal=0

    for i in range(0, len(portfolio)-1):


        if portfolio.index[i] in allocation.index:

            j = assets_data.index.get_loc(portfolio.index[i + 1])
            k = allocation.index.get_loc(portfolio.index[i])
            i_rebal = portfolio.index.get_loc(portfolio.index[i])
            j_rebal = assets_data.index.get_loc(portfolio.index[i])

            portfolio[i + 1] = portfolio[i_rebal]*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()

        else:

            j = assets_data.index.get_loc(portfolio.index[i + 1])

            portfolio[i + 1] = portfolio[i_rebal]*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()

    return portfolio


