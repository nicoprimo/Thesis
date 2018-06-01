import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read LV and MV aggregated demand // PV production // Price of electricity from the grid
# PV production
df_PV = pd.read_csv('pv_production.csv', index_col=0)
df_PV['time'] = pd.to_datetime(df_PV.index, dayfirst=True)
df_PV.index = df_PV['time']

# get he data for the reference weeks
df_PV_march = df_PV.loc['20160314':'20160321000000']
df_PV_june = df_PV.loc['20160606':'20160613000000']
df_PV_september = df_PV.loc['20160912':'20160919000000']
df_PV_december = df_PV.loc['20161205':'20161212000000']

pv_production_march = df_PV_march['PV production']
pv_production_june = df_PV_june['PV production']
pv_production_september = df_PV_september['PV production']
pv_production_december = df_PV_december['PV production']

pv_production = pd.concat([pv_production_march,
                           pv_production_june,
                           pv_production_september,
                           pv_production_december],
                          ignore_index=True)

# Demand
df_LV = pd.read_csv('community_demand.csv', index_col=0)
df_MV = pd.read_csv('MV_demand.csv', index_col=0)
df_MV['time'] = pd.to_datetime(df_MV.index, dayfirst=True)
df_MV.index = df_MV['time']

df_MV_march = df_MV.loc['20160314':'20160321']
df_MV_june = df_MV.loc['20160606':'20160613']
df_MV_september = df_MV.loc['20160912':'20160919']
df_MV_december = df_MV.loc['20161205':'20161212']

LV_consumption_winter = np.asarray(df_LV['final consumption winter'])
LV_consumption_summer = np.asarray(df_LV['final consumption winter'])   # need to create the summer consumption for LV
MV_consumption_march = np.asarray(df_MV_march['final consumption march'])
MV_consumption_june = np.asarray(df_MV_june['final consumption june'])
MV_consumption_september = np.asarray(df_MV_september['final consumption september'])
MV_consumption_december = np.asarray(df_MV_december['final consumption december'])

consumption_march = pd.Series(LV_consumption_winter + MV_consumption_march)
consumption_june = pd.Series(LV_consumption_summer + MV_consumption_june)
consumption_september = pd.Series(LV_consumption_summer + MV_consumption_september)
consumption_december = pd.Series(LV_consumption_winter + MV_consumption_december)

demand = pd.concat([consumption_march,
                    consumption_june,
                    consumption_september,
                    consumption_december],
                   ignore_index=True)

# Grid price
df_tariff_winter = pd.read_excel('tariff.xlsx', sheetname='Winter_week')
df_tariff_summer = pd.read_excel('tariff.xlsx', sheetname='Summer_week')

grid_price_winter = df_tariff_winter['Price']
grid_price_summer = df_tariff_summer['Price']

grid_price = pd.concat([grid_price_winter,
                        grid_price_summer,
                        grid_price_summer,
                        grid_price_winter],
                       ignore_index=True)


# Need to get real value for PV system
LC = 100 / 52 * 4    # Price in euro/kW of PV installed per 1 week (1 year / 52 weeks) * 4 weeks (reference ones)

# Start iteration to see the optimal number of PV to be installed
# Get the price of Scenario 1 - No PV installed
cost1 = 0
for i in range(672 * 4):
    cost1 = cost1 + demand[i] * grid_price[i]

# Set up for the Scenario 2 - PV installed
feed_in_tariff = 0  # should be set as the average price in the stock market times 0.90
n_set = 1
min_cost = cost1

cost2 = [0 for x in range(100)]     # to keep track of the price variation

for j in range(100):    # See the price variations up to 100 PV systems
    PV = pv_production + j * pv_production
    for i in range(672 * 4):
        if demand[i] > PV[i]:
            cost2[j] = cost2[j] + (demand[i] - PV[i]) * grid_price[i]
        elif demand[i] <= PV[i]:
            cost2[j] = cost2[j] + (demand[i] - PV[i]) * feed_in_tariff

    cost2[j] = cost2[j] + (j + 1) * 5 * LC
    if cost2[j] < min_cost:
        min_cost = cost2[j]
        n_set = j + 1

max_power = demand.max() / .25
print(max_power)
print(n_set)
print(cost1)
print(min(cost2))

plt.plot(demand)
plt.plot(pv_production * n_set)
plt.show()

plt.plot(cost2)
plt.show()
