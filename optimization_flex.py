import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Variable
number_ewh = 50
LC = 96.65 / 52 * 4  # Price in euro/kW of PV installed per 1 week (1 year = 52 weeks) * 4 weeks (reference ones)
feed_in_tariff = 0.0414  # should be set as the average price in the stock market times 0.90 - 0.0377 from literature

gap = 150

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
LV_consumption_summer = np.asarray(df_LV['final consumption winter'])  # need to create the summer consumption for LV
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
df_tariff_winter = pd.read_excel('tariff.xlsx', sheet_name='Winter_week')
df_tariff_summer = pd.read_excel('tariff.xlsx', sheet_name='Summer_week')

grid_price_winter = df_tariff_winter['Price']
period_price_winter = df_tariff_winter['Period']
grid_price_summer = df_tariff_summer['Price']
period_price_summer = df_tariff_summer['Period']

grid_price = pd.concat([grid_price_winter,
                        grid_price_summer,
                        grid_price_summer,
                        grid_price_winter], ignore_index=True)

period_price = pd.concat([period_price_winter,
                          period_price_summer,
                          period_price_summer,
                          period_price_winter],
                         ignore_index=True)


# Get profiles on/of for EWH

ewh_profiles = pd.read_csv('ewh_%d.csv' % number_ewh, index_col=[0, 1])
ewh_consumption = 4.2 * .25  # Nominal Power = 4.2 kW

# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'flex': ewh_profiles.groupby(level=1)['consumption'].sum(),
                   'pv production': pv_production,
                   'grid price': grid_price}, dtype=float)

df['period price'] = period_price

inputs = pd.DataFrame({'EWH': df['flex'],
                       'LV': np.tile(LV_consumption_winter, 4),
                       'MV': pd.concat([pd.Series(MV_consumption_march),
                                       pd.Series(MV_consumption_june),
                                       pd.Series(MV_consumption_september),
                                       pd.Series(MV_consumption_december)], ignore_index=True),
                       'Tariff': period_price})

# Definition of functions


def cost_period(demand, flex, pv_production, grid_price, n_set):
    if (demand + flex - n_set * pv_production) > 0:
        return (demand + flex - n_set * pv_production) * grid_price
    else:
        return (demand + flex - n_set * pv_production) * feed_in_tariff


def self_consumption(demand, flex, pv_production, n_set):
    if (demand + flex - n_set * pv_production) >= 0:
        return n_set * pv_production
    else:
        return demand + flex


def sun_surplus(demand, flex, pv_production, n_set):  # function to get the value of ewh
    # that can be shift in a specific time frame
    surplus = int((n_set * pv_production - demand + flex) / ewh_consumption)
    if surplus < 0:
        surplus = 0
    return surplus


# vectorize function to pass arrays instead of single values
v_cost_period = np.vectorize(cost_period)
v_sun_surplus = np.vectorize(sun_surplus)
v_self_consumption = np.vectorize(self_consumption)


# Scenario 2
cost2 = np.zeros(gap)  # to keep track of the price variation
E_SC_2 = np.zeros(gap)
SSR_2 = np.zeros(gap)
SSR_ewh_2 = np.zeros(gap)
SCR_2 = np.zeros(gap)
SCR_ewh_2 = np.zeros(gap)

for n_set2 in range(gap):  # See the price variations up to 100 PV systems
    df['cost2'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set2)
    cost2[n_set2] = df['cost2'].sum() + n_set2 * 5 * LC

    E_SC_2[n_set2] = v_self_consumption(df['demand'].values, df['flex'].values,
                                        df['pv production'].values, n_set2).sum()
    SSR_2[n_set2] = (v_self_consumption(df['demand'].values, df['flex'].values,
                     df['pv production'].values, n_set2).sum() / (df['demand'].sum() + df['flex'].sum())) * 100
    SSR_ewh_2[n_set2] = (v_self_consumption(0, df['flex'].values,
                         df['pv production'].values, n_set2).sum() / (df['flex'].sum())) * 100
    SCR_2[n_set2] = (v_self_consumption(df['demand'].values, df['flex'].values,
                     df['pv production'].values, n_set2).sum() / (df['pv production'] * n_set2).sum()) * 100
    SCR_ewh_2[n_set2] = (v_self_consumption(0, df['flex'].values, df['pv production'].values, n_set2).sum() / (
                                     df['pv production'] * n_set2).sum()) * 100


# Scenario 3

cost3 = np.zeros(gap)
E_SC_3 = np.zeros(gap)
SSR_3 = np.zeros(gap)
SSR_ewh_3 = np.zeros(gap)
SCR_3 = np.zeros(gap)
SCR_ewh_3 = np.zeros(gap)

for n_set3 in range(gap):
    surplus = v_sun_surplus(df['demand'], df['flex'], df['pv production'], n_set3)
    ewh_profiles_copy = ewh_profiles.copy()
    for ewh in range(number_ewh):
        total_status_on = len(ewh_profiles.loc[ewh, 'shower'].dropna())
        indices_list = ewh_profiles.loc[ewh, 'shower'].dropna().index
        for i in range(total_status_on - 1):
            initial_on_status = indices_list[i]
            end_on_status = indices_list[i + 1]
            # check if there is a surplus between the two indices
            if surplus[initial_on_status:end_on_status].max() > 0:
                shift_relative_position = surplus[initial_on_status:end_on_status].argmax()
                real_position = shift_relative_position + initial_on_status
                # shift the initial status

                ewh_profiles_copy.loc[ewh, 'consumption'][initial_on_status + 1] = 0
                ewh_profiles_copy.loc[ewh, 'consumption'][initial_on_status + 2] = 0

                ewh_profiles_copy.loc[ewh, 'consumption'][real_position] = ewh_consumption
                ewh_profiles_copy.loc[ewh, 'consumption'][real_position + 1] = ewh_consumption

                surplus[real_position] -= 1
                surplus[real_position + 1] -= 1

    df['new flex'] = ewh_profiles_copy.groupby(level=1)['consumption'].sum()

    df['cost3'] = v_cost_period(df['demand'].values, df['new flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set3)
    cost3[n_set3] = df['cost3'].sum() + n_set3 * 5 * LC

    if n_set3 == 10:
        inputs['EWH flex 50'] = df['new flex']
    if n_set3 == 20:    # Scenario with 100 kW PV capacity
        inputs['EWH flex 100'] = df['new flex']
    if n_set3 == 30:    # Scenario with 150 kW PV capacity
        inputs['EWH flex 150'] = df['new flex']
    if n_set3 == 40:
        inputs['EWH flex 200'] = df['new flex']
    if n_set3 == 50:    # Scenario with 250 kW PV capacity
        inputs['EWH flex 250'] = df['new flex']
    if n_set3 == 60:
        inputs['EWH flex 300'] = df['new flex']
    if n_set3 == 70:    # Scenario with 350 kW PV capacity
        inputs['EWH flex 350'] = df['new flex']
    if n_set3 == 80:
        inputs['EWH flex 400'] = df['new flex']
    if n_set3 == 90:
        inputs['EWH flex 450'] = df['new flex']
    if n_set3 == 100:
        inputs['EWH flex 500'] = df['new flex']
    if n_set3 == 110:
        inputs['EWH flex 550'] = df['new flex']
    if n_set3 == 120:
        inputs['EWH flex 600'] = df['new flex']
    if n_set3 == 130:
        inputs['EWH flex 650'] = df['new flex']
    if n_set3 == 140:
        inputs['EWH flex 700'] = df['new flex']

    if n_set3 == cost2.argmin():
            df['new flex1'] = df['new flex']
            df['sun surplus'] = v_sun_surplus(df['demand'], df['flex'], df['pv production'], n_set3)

    E_SC_3[n_set3] = v_self_consumption(df['demand'].values, df['new flex'].values,
                                        df['pv production'].values, n_set3).sum()
    SSR_3[n_set3] = (v_self_consumption(df['demand'].values, df['new flex'].values,
                                        df['pv production'].values, n_set3).sum() / (
                                 df['demand'].sum() + df['new flex'].sum())) * 100
    SSR_ewh_3[n_set3] = (v_self_consumption(0, df['new flex'].values,
                                            df['pv production'].values, n_set3).sum() / (df['new flex'].sum())) * 100
    SCR_3[n_set3] = (v_self_consumption(df['demand'].values, df['new flex'].values,
                                        df['pv production'].values, n_set3).sum() / (
                                 df['pv production'] * n_set3).sum()) * 100
    SCR_ewh_3[n_set3] = (v_self_consumption(0, df['new flex'].values, df['pv production'].values, n_set3).sum() / (
            df['pv production'] * n_set3).sum()) * 100


# Print results
outcomes = pd.DataFrame({'cost2': cost2,
                         'E SC2': E_SC_2,
                         'SCR2': SCR_2,
                         'SSR2': SSR_2,
                         'SSR2 ewh': SSR_ewh_2,
                         'cost3': cost3,
                         'E SC3': E_SC_3,
                         'SCR3': SCR_3,
                         'SSR3': SSR_3,
                         'SSR3 ewh': SSR_ewh_3})


inputs['EWH flex'] = df['new flex1']
inputs['PV'] = df['pv production'].values
inputs.to_csv('inputs1_%d.csv' % number_ewh)
outcomes.to_csv('outcomes1_%d.csv' % number_ewh)

print('* Set up *')
print('FIT = %.2f' % feed_in_tariff + '€/kWh')
print('LC = %.2f' % (LC*52/4) + '€/kW/year')
print('number of EWHs = %d' % number_ewh)
print('* Yearly Costs *')
print('without %.2f' % min(cost2))
print('with %.2f' % min(cost3))
print('* Optimal PV installation *')
print('without %d' % (cost2.argmin() * 5) + 'kW')
print('with %d' % (cost3.argmin() * 5) + 'kW')
print('* Self Consumption comparison *')
print('SCR')
print('without flex %.2f' % SCR_2[cost2.argmin()])
print('with flex %.2f' % SCR_3[cost2.argmin()])
print('SSR')
print('without flex %.2f' % SSR_2[cost2.argmin()])
print('with flex %.2f' % SSR_3[cost2.argmin()])
print('SCR only EWH')
print('without flex %.2f' % SCR_ewh_2[cost2.argmin()])
print('with flex %.2f' % SCR_ewh_3[cost2.argmin()])
print('SSR only EWH')
print('without flex %.2f' % SSR_ewh_2[cost2.argmin()])
print('with flex %.2f' % SSR_ewh_3[cost2.argmin()])

plt.plot(df['pv production'] * (cost3.argmin()), 'r')
plt.plot(df['demand'] + df['new flex1'], '--')
plt.grid()
plt.show()

plt.plot(df['sun surplus'], 'y')
plt.plot(df['flex']/ewh_consumption, 'g')
plt.plot(df['new flex1']/ewh_consumption, '--k')
plt.show()

# Scenario 2 vs Scenario 3 - overall
fig, ax1 = plt.subplots()
ax1.set_xlabel('number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)


ax2 = ax1.twinx()
ax2.set_ylabel('Self Sufficiency Rates (%) ')
ax2.plot(SSR_2, 'y', label='SSR2')
ax2.plot(SSR_3, '-.k', label='SSR3')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
# -----------------------------------
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)


ax2 = ax1.twinx()
ax2.set_ylabel('Self Consumption Rates (%) ')
ax2.plot(SCR_2, 'y', label='SCR2')
ax2.plot(SCR_3, '-.k', label='SCR3')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()

# Scenario 2 and 3 EWH comparison
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.set_ylabel('Self Sufficiency Ratio (%)')
ax2.plot(SSR_ewh_2, 'y', label='SSR2 only EWH')
ax2.plot(SSR_ewh_3, '-.k', label='SSR3 only EWH')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
