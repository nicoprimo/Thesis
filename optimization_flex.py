import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# Create flex consumption - EWH rated power = 4.5 kW
df_flex = pd.DataFrame({'consumption': [0 for z in range(96)]}, index=pd.date_range('00:15', periods=96, freq='15min'))
flex_consumption = df_flex['consumption']
rated_power = 4.5
flex_consumption_single = rated_power * .25  # consumption per period in kWh
tot_ewh = 50        # variable

# Create normal distribution function to set up how many EWH are on during the day - NOT WORKING!
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 48)
gauss = pd.Series(norm.pdf(x))
gauss = gauss.append(gauss, ignore_index=True)

number_ewh_on = pd.Series(gauss * tot_ewh).round()
flex_consumption[0] = (number_ewh_on[0] + number_ewh_on[95] + number_ewh_on[94]) * flex_consumption_single
flex_consumption[1] = (number_ewh_on[1] + number_ewh_on[0] + number_ewh_on[95]) * flex_consumption_single

for i in range(2, 96):
    flex_consumption[i] = (number_ewh_on[i] + number_ewh_on[i-1] + number_ewh_on[i-1]) * flex_consumption_single

flex_consumption = pd.concat(4 * 7 * [flex_consumption],
                             ignore_index=True)


demand_with_flex = np.asarray(demand) + np.asarray(flex_consumption)
plt.plot(demand_with_flex)
plt.plot(demand, 'r')
plt.show()
