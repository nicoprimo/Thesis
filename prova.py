import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Read LV and MV aggregated demand // PV production // Price of electricity from the grid
# PV production
df_PV = pd.read_csv('pv_production.csv', index_col=0)
df_PV['time'] = pd.to_datetime(df_PV.index, dayfirst=True)
df_PV.index = df_PV['time']

# get he data for the reference weeks
df_PV_march = df_PV.loc['20160314':'201603210001']
df_PV_june = df_PV.loc['201606060001':'201606130001']
df_PV_september = df_PV.loc['201609120001':'201609190001']
df_PV_december = df_PV.loc['201612050001':'201612120001']

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

df_MV_march = df_MV.loc['20160314':'201603210001']
df_MV_june = df_MV.loc['201606060001':'201606130001']
df_MV_september = df_MV.loc['201609120001':'201609190001']
df_MV_december = df_MV.loc['201612050001':'201612120001']

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

# Read flex consumption - EWH rated power = 4.5 kW
flex = pd.concat(4 * [df_LV['flex winter']], ignore_index=True)     # get the consumption from flexible assets
flex1 = flex
ewh_status = pd.concat(4 * [df_LV['63']], ignore_index=True)        # get the status of clustered ewh
ewh_n_status = pd.concat(4 * [df_LV.drop(df_LV.columns[[0, 1, 65]], axis=1)], ignore_index=True)
number_ewh = 50
ewh_n_status = ewh_n_status.iloc[:, :number_ewh]     # get only the desired profiles

demand_flex = pd.Series(np.asarray(flex) + np.asarray(demand))
rated_power = 4.5
ewh_consumption_single = rated_power * .25  # consumption per period in kWh

# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'flex': flex,
                   'pv production': pv_production,
                   'ewh status': ewh_status,
                   'grid price': grid_price}, dtype=float)
df = pd.concat([df, ewh_n_status], axis=1)

# Need to get real value for PV system
LC = 100 / 52 * 4       # Price in euro/kW of PV installed per 1 week (1 year / 52 weeks) * 4 weeks (reference ones)
feed_in_tariff = 0      # should be set as the average price in the stock market times 0.90

# Start iteration to see the optimal number of PV to be installed
# Get the price of Scenario 1 - No PV installed

df['cost1'] = df.apply(lambda row: (row['demand'] + row['flex']) * row['grid price'], axis=1)
cost1 = df['cost1'].sum()

# Set up for the Scenario 2 - PV installed


def cost_period(demand, flex, pv_production, grid_price, n_set):
    if (demand + flex - n_set * pv_production) > 0:
        return (demand + flex - n_set * pv_production) * grid_price
    else:
        return (demand + flex - n_set * pv_production) * feed_in_tariff


# vectorize function to pass arrays instead of single values
v_cost_period = np.vectorize(cost_period)
cost2 = np.zeros(100)     # to keep track of the price variation
for n_set2 in range(0, 100):    # See the price variations up to 100 PV systems
    df['cost2'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set2)
    cost2[n_set2] = df['cost2'].sum() + n_set2 * 5 * LC

# Set up Scenario 3 optimization with flexibility


def sun_surplus(demand, flex, pv_production, n_set):    # function to get the value of ewh
    # that can be shift in a specific time frame
    return (demand + flex - n_set * pv_production) / ewh_consumption_single


# vectorize sun_surplus
v_sun_surplus = np.vectorize(sun_surplus)
cost3 = np.zeros(50)
# n_set3 range close to the optimal n_set2 (cost2.argmin()) // Expected n_set3 > n_set2
for n_set3 in range(cost2.argmin()-20, cost2.argmin()+30):    # See the price variations up to 100 PV systems
    # add flexibility part // Align as much as possible ewh status and sun surplus //
    # work on ewh_status and sun_surplus
    df['sun surplus'] = v_sun_surplus(df['demand'], df['flex'], df['pv_production'], n_set3)
    # recalculate flex vector
    df['flex'] = df['ewh status'].values * ewh_consumption_single
    # Get the cost for the scenario
    df['cost3'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set3)
    cost2[n_set3] = df['cost3'].sum() + n_set3 * 5 * LC

print(cost2.min())
print(cost2.argmin())

plt.plot(cost2)
plt.grid()
plt.show()
