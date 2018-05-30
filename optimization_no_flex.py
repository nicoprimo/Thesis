import pandas as pd
import matplotlib.pyplot as plt

# Read LV and MV aggregated demand // PV production // Price of electricity from the grid
# PV production
df_PV = pd.read_csv('pv_production.csv', index_col=0)
df_PV['time'] = pd.to_datetime(df_PV.index, dayfirst=True)
df_PV.index = df_PV['time']
# get he data just of the week of March
df_PV1 = df_PV.loc['20160314':'20160321000000']
pv_production = df_PV1['PV production']

# Demand
df_LV = pd.read_csv('community_demand.csv', index_col=0)
df_MV = pd.read_csv('MV_demand.csv', index_col=0)
df_LV['Demand'] = df_MV['final consumption'] + df_MV['final consumption']
demand = df_LV['Demand']
demand.index = df_PV1.index

# Grid price
df_tariff = pd.read_excel('tariff.xlsx', sheetname='Winter_week')
grid_price_winter = df_tariff['Price']
grid_price_winter.index = df_PV1.index

# Need to get real value for PV system
LC = 100 / 52     # Price in euro/kW of PV installed per 1 week (1 year / 52 weeks)

# Start iteration to see the optimal number of PV to be installed
# Get the price of Scenario 1 - No PV installed
cost1 = 0
for i in range(672):
    cost1 = cost1 + demand[i] * grid_price_winter[i]

# Set up for the Scenario 2 - PV installed
feed_in_tariff = 0  # should be set as the average price in the stock market times 0.90
n_set = 1
min_cost = cost1

cost2 = [0 for x in range(100)]     # to keep track of the price variation

for j in range(100):    # See the price variations up to 100 PV systems
    PV = pv_production + j * pv_production
    for i in range(672):
        if demand[i] > PV[i]:
            cost2[j] = cost2[j] + ( demand[i] - PV[i]) * grid_price_winter[i]
        elif demand[i] <= PV[i]:
            cost2[j] = cost2[j] + (demand[i] - PV[i]) * feed_in_tariff

    cost2[j] = cost2[j] + (j + 1) * 5 * LC
    if cost2[j] < min_cost:
        min_cost = cost2[j]
        n_set = j + 1

max_power = demand.max() / .25
print(max_power)
plt.plot(df_LV['final consumption'])
plt.plot(df_MV['final consumption'])
plt.show()

plt.plot(n_set * pv_production)
plt.plot(demand)
plt.show()
