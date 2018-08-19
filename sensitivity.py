import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Variable
LC = 96.65 / 52 * 4  # Price in euro/kW of PV installed per 1 week (1 year = 52 weeks) * 4 weeks (reference ones)
feed_in_tariff = 0.0377  # should be set as the average price in the stock market times 0.90 - 0.0377 from literature
range_gap = 20
n_set = 70      # 420 kW installed

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


# Generate profiles on/of for EWH deciding morning/evening peak hours and peak gap


def assign_with_probability(p):
    return np.random.choice(a=[1, 0], p=[p, 1 - p])


# Vectorize functions
v_assign_with_probability = np.vectorize(assign_with_probability)


def ewh_profile(peak_gap, morning_peak, evening_peak):
    # Create a vector with very low probability
    index_minutes = pd.date_range(start='22/06/2018', periods=24*60, freq='T')
    probability_vector = np.full(24 * 60, fill_value=0.001)
    probability_vector[(morning_peak * 60) - peak_gap:(morning_peak * 60) + peak_gap] = \
        norm.pdf(np.linspace(norm.ppf(0.001),
                             norm.ppf(0.999), peak_gap * 2), scale=1)
    probability_vector[(evening_peak * 60) - peak_gap:(evening_peak * 60) + peak_gap] = \
        norm.pdf(np.linspace(norm.ppf(0.001),
                             norm.ppf(0.999), peak_gap * 2), scale=1)
    probability_vector = pd.Series(probability_vector, index=index_minutes)
    probability_vector = probability_vector.resample('15T').mean()
    # Create a probability vector for 4 weeks
    probability_vector_week = np.tile(probability_vector.values, 7 * 4)

    # Create vector ewh_on vector using probability_vector
    return v_assign_with_probability(probability_vector_week)


# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'pv production': pv_production,
                   'grid price': grid_price}, dtype=float)

# assign Nan to all the time steps where there are no EWHs on - Easy later
df['period price'] = period_price

# Decide rated power, therefore single consumption per EWH on
rated_power = 4.2
ewh_consumption = rated_power * .25  # consumption per period in kWh
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


# Start sensitivity - PV systems install fixed, varying number of EWHs

cost3 = np.zeros(range_gap)
SSR_3 = np.zeros(range_gap)
SSR_ewh_3 = np.zeros(range_gap)
SCR_3 = np.zeros(range_gap)
SCR_ewh_3 = np.zeros(range_gap)

cost2 = np.zeros(range_gap)  # to keep track of the price variation
SSR_2 = np.zeros(range_gap)
SSR_ewh_2 = np.zeros(range_gap)
SCR_2 = np.zeros(range_gap)
SCR_ewh_2 = np.zeros(range_gap)

cost3_min = cost2.min()

for gap in range(1, range_gap+1):
    print('**** EWHs %d ****' % (gap*10))
    # Calculate the flex part of the demand
    number_ewh = gap * 10

    #  Create MultiIndex DataFrame for EWH profiles
    iterable = [np.array(range(number_ewh)),
                np.array(range(2688))]
    index = pd.MultiIndex.from_product(iterable)
    ewh_profiles = pd.DataFrame(index=index,
                                columns=['shower', 'consumption'])

    for i in range(number_ewh):
        ewh_profiles.loc[i, 'shower'] = ewh_profile(300, 8, 19)

        for day in range(7 * 4):
            # Clean morning (from 0 till 47)
            n_drop = ewh_profiles.loc[i, 'shower'][day * 96:(day * 96 + 48)][
                         ewh_profiles.loc[i, 'shower'] > 0].count() - 1

            if n_drop > 0:
                drop_indices = np.random.choice(ewh_profiles.loc[i, 'shower'][day * 96:(day * 96 + 48)]
                                                [ewh_profiles.loc[i, 'shower'] > 0].index, n_drop, replace=False)
                ewh_profiles.loc[i, 'shower'][drop_indices] = 0

            consumption_time = ewh_profiles.loc[i, 'shower'][day * 96:(day * 96 + 48)][
                ewh_profiles.loc[i, 'shower'] > 0]

            #  ewh_profiles.loc[i, 'shower'][day*96:(day*96+47)][ewh_profiles.loc[i, 'shower'] == 0] = np.nan
            ewh_profiles.loc[i, 'consumption'][consumption_time.index + 1] = ewh_consumption
            ewh_profiles.loc[i, 'consumption'][consumption_time.index + 2] = ewh_consumption

            # Clean evening (from 48 till 95)
            n_drop = ewh_profiles.loc[i, 'shower'][(day * 96 + 48):(day * 96 + 96)][
                         ewh_profiles.loc[i, 'shower'] > 0].count() - 1
            if n_drop > 0:
                drop_indices = np.random.choice(ewh_profiles.loc[i, 'shower'][(day * 96 + 48):(day * 96 + 96)]
                                                [ewh_profiles.loc[i, 'shower'] > 0].index, n_drop, replace=False)
                ewh_profiles.loc[i, 'shower'][drop_indices] = 0

            consumption_time = ewh_profiles.loc[i, 'shower'][(day * 96 + 48):(day * 96 + 96)][
                ewh_profiles.loc[i, 'shower'] > 0]
            if consumption_time.index >= 2685:
                ewh_profiles.loc[i, 'consumption'][2686] = ewh_consumption
                ewh_profiles.loc[i, 'consumption'][2867] = ewh_consumption

                ewh_profiles.loc[i, 'shower'][2685:2688] = 0
                ewh_profiles.loc[i, 'shower'][2685] = 1
            else:
                ewh_profiles.loc[i, 'consumption'][consumption_time.index + 1] = ewh_consumption
                ewh_profiles.loc[i, 'consumption'][consumption_time.index + 2] = ewh_consumption

        ewh_profiles.loc[i, 'shower'].replace(0, np.nan, inplace=True)
        ewh_profiles.loc[i, 'consumption'].fillna(value=0, inplace=True)

    df['flex'] = ewh_profiles.groupby(level=1)['consumption'].sum()
    df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, n_set)

    # Scenario 2
    df['cost2'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set)
    cost2[gap-1] = df['cost2'].sum() + n_set * 5 * LC
    SSR_2[gap-1] = (v_self_consumption(df['demand'].values, df['flex'].values,
                                        df['pv production'].values, n_set).sum() / (
                                 df['demand'].sum() + df['flex'].sum())) * 100
    SSR_ewh_2[gap-1] = (v_self_consumption(0, df['flex'].values,
                                            df['pv production'].values, n_set).sum() / (df['flex'].sum())) * 100
    SCR_2[gap-1] = (v_self_consumption(df['demand'].values, df['flex'].values,
                                        df['pv production'].values, n_set).sum() / (
                                 df['pv production'] * n_set).sum()) * 100
    SCR_ewh_2[gap-1] = (v_self_consumption(0, df['flex'].values, df['pv production'].values, n_set).sum() / (
            df['pv production'] * n_set).sum()) * 100

    # Scenario 3
    surplus = v_sun_surplus(df['demand'], df['flex'], df['pv production'], n_set)

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
                                df['pv production'].values, df['grid price'].values, n_set)
    cost3[gap-1] = df['cost3'].sum() + n_set * 5 * LC

    SSR_3[gap-1] = (v_self_consumption(df['demand'].values, df['new flex'].values,
                                        df['pv production'].values, n_set).sum() / (
                             df['demand'].sum() + df['new flex'].sum())) * 100
    SSR_ewh_3[gap-1] = (v_self_consumption(0, df['new flex'].values,
                                            df['pv production'].values, n_set).sum() / (df['new flex'].sum())) * 100
    SCR_3[gap-1] = (v_self_consumption(df['demand'].values, df['new flex'].values,
                                        df['pv production'].values, n_set).sum() / (
                             df['pv production'] * n_set).sum()) * 100
    SCR_ewh_3[gap-1] = (v_self_consumption(0, df['new flex'].values, df['pv production'].values, n_set).sum() / (
            df['pv production'] * n_set).sum()) * 100


# Print results
print('* Set up *')
print('FIT = %.2f' % feed_in_tariff + '€/kWh')
print('LC = %.2f' % (LC*52/4) + '€/kW/year')
print('PV panels installed = %d' % (n_set*5) + 'kW')


# Scenario 2 vs Scenario 3 - Overall
fig, ax1 = plt.subplots()
ax1.set_xlabel('number of EWHs (x10)')
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
ax1.set_xlabel('Number of EWHs (x10)')
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
ax1.set_xlabel('Number of EWHs (x10)')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(SSR_ewh_2, 'y', label='SSR2 only EWH')
ax2.plot(SSR_ewh_3, '-.k', label='SSR3 only EWH')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
# ----------------------------
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of EWHs (x10)')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.set_ylabel('Self Consumption Ratio (%)')
ax2.plot(SCR_ewh_2, 'y', label='SCR2 only EWHs')
ax2.plot(SCR_ewh_3, '-.k', label='SCR3 only EWHs')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
