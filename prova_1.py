import pandas as pd
import numpy as np
from math import exp, pi, sqrt
import matplotlib.pyplot as plt

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
df_tariff_winter = pd.read_excel('tariff.xlsx', sheetname='Winter_week')
df_tariff_summer = pd.read_excel('tariff.xlsx', sheetname='Summer_week')

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


def gauss(x):
    return exp(-x ** 2 / 2) / sqrt(2 * pi)


def assign_with_probability(p):
    return np.random.choice(a=[1, 0], p=[p, 1 - p])


# Vectorize functions
v_gauss = np.vectorize(gauss)
v_assign_with_probability = np.vectorize(assign_with_probability)


def ewh_profile(peak_gap, morning_peak, evening_peak):
    # Create a vector with very low probability
    probability_vector = np.full(96, fill_value=0.02)
    probability_vector[(morning_peak * 4) - peak_gap:(morning_peak * 4) + peak_gap] = \
        v_gauss(np.array(range(-peak_gap, peak_gap)))
    probability_vector[(evening_peak * 4) - peak_gap:(evening_peak * 4) + peak_gap] = \
        v_gauss(np.array(range(-peak_gap, peak_gap)))
    # Create a probability vector for 4 weeks
    probability_vector_week = np.tile(probability_vector, 7 * 4)
    # Create vector ewh_on vector using probability_vector
    return v_assign_with_probability(probability_vector_week)


## This last step could be improved! ##
number_ewh = 100
ewh_profiles = pd.DataFrame()
for i in range(number_ewh):
    ewh_profiles['%d' % i] = ewh_profile(10, 8, 20)

# Sum how many EWH are on per period
ewh_profiles['total'] = ewh_profiles.sum(1)

# Decide rated power, therefore single consumption per EWH on
rated_power = 4.5
ewh_consumption_single = rated_power * .25  # consumption per period in kWh

# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'flex': ewh_profiles['total'].values * ewh_consumption_single,
                   'pv production': pv_production,
                   'ewh status': ewh_profiles['total'].values,
                   'grid price': grid_price}, dtype=float)
ewh_profiles.drop('total', axis=1, inplace=True)
df = pd.concat([df, ewh_profiles], axis=1)
df['period price'] = period_price

# Need to get real value for PV system
LC = 100 / 52 * 4  # Price in euro/kW of PV installed per 1 week (1 year = 52 weeks) * 4 weeks (reference ones)

feed_in_tariff = 0.03  # should be set as the average price in the stock market times 0.90


# Definition of functions


def cost_period(demand, flex, pv_production, grid_price, n_set):
    if (demand + flex - n_set * pv_production) > 0:
        return (demand + flex - n_set * pv_production) * grid_price
    else:
        return (demand + flex - n_set * pv_production) * feed_in_tariff


def sun_surplus(demand, flex, pv_production, n_set):  # function to get the value of ewh
    # that can be shift in a specific time frame
    surplus = int((n_set * pv_production - demand + flex) / ewh_consumption_single)
    if surplus < 0:
        surplus = 0
    return surplus


# vectorize function to pass arrays instead of single values
v_cost_period = np.vectorize(cost_period)
v_sun_surplus = np.vectorize(sun_surplus)

cost2 = np.zeros(100)  # to keep track of the price variation
for n_set2 in range(0, 100):  # See the price variations up to 100 PV systems
    df['cost2'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set2)
    cost2[n_set2] = df['cost2'].sum() + n_set2 * 5 * LC

cost3 = np.zeros(100)
for n_set3 in range(1, 100):  # See the price variations up to 100 PV systems
    print('Number of PV %d' % n_set3)
    # add flexibility part // Align as much as possible ewh status and sun surplus //
    df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, n_set3)

    new_ewh_status = np.zeros(shape=(96, 28, number_ewh + 1))
    for day in range(28):
        print('DAY %d' % day)
        # Get the values of the day
        df_day = df[(day * 96):((day + 1) * 96)]
        df_day.reset_index(inplace=True, drop=True)

        period_price_day = df_day['period price'].values
        sun_surplus_day = df_day['sun surplus'].values

        ewh_status_day = df_day['ewh status'].values
        new_ewh_status_day = new_ewh_status[:, day, :]
        ewh_n_status_day = ewh_profiles[(day * 96):((day + 1) * 96)]
        ewh_n_status_day.reset_index(inplace=True, drop=True)

        # Get the profiles "ponta"
        ewh_ponta_day = ewh_n_status_day[df_day['period price'] == 'p'].where(ewh_n_status_day > 0)
        ewh_ponta_day.dropna(axis=1, how='all', inplace=True)
        ewh_ponta_day.dropna(axis=0, how='all', inplace=True)
        if not ewh_ponta_day.empty:
            ponta_starting_index = ewh_ponta_day.index[0]

        # Get the profiles "cheia"
        ewh_cheia_day = ewh_n_status_day[df_day['period price'] == 'c'].where(ewh_n_status_day > 0)
        ewh_cheia_day.dropna(axis=1, how='all', inplace=True)
        ewh_cheia_day.dropna(axis=0, how='all', inplace=True)
        if not ewh_cheia_day.empty:
            cheia_starting_index = ewh_cheia_day.index[0]

        # Get the profiles "vazio"
        ewh_vazio_day = ewh_n_status_day[df_day['period price'] == 'v'].where(ewh_n_status_day > 0)
        ewh_vazio_day.dropna(axis=1, how='all', inplace=True)
        ewh_vazio_day.dropna(axis=0, how='all', inplace=True)
        if not ewh_vazio_day.empty:
            vazio_starting_index = ewh_vazio_day.index[0]

        # Get the profiles "super vazio"
        ewh_supervazio_day = ewh_n_status_day[df_day['period price'] == 'sv'].where(ewh_n_status_day > 0)
        ewh_supervazio_day.dropna(axis=1, how='all', inplace=True)
        ewh_supervazio_day.dropna(axis=0, how='all', inplace=True)
        if not ewh_vazio_day.empty:
            supervazio_starting_index = ewh_supervazio_day.index[0]

        for time_step in range(96):
            # Fill up sun surplus
            while sun_surplus_day[time_step] > 0 and (not ewh_ponta_day.empty or not ewh_cheia_day.empty
                                                      or not ewh_vazio_day.empty or not ewh_supervazio_day.empty):
                # Check if there is any ponta profile before time_step
                if not ewh_ponta_day.empty:
                    # Get how many EWH profiles are available
                    profile_available = ewh_ponta_day.head(1).dropna(axis=1).shape[1]
                    # Check if there is the need to shift all the profiles
                    if profile_available > sun_surplus_day[time_step]:
                        profile_available = sun_surplus_day[time_step]
                    # Shift the profiles available to the sun surplus
                    for i in range(profile_available):
                        ewh_available = int(ewh_ponta_day.head(1).dropna(axis=1).columns[i])
                        new_ewh_status_day[time_step, ewh_available] += 1
                        new_ewh_status_day[ponta_starting_index, ewh_available] -= 1
                    # Update index of first values available and drop the 1st row of profiles used
                    sun_surplus_day[time_step] -= profile_available
                    ewh_ponta_day.drop([ponta_starting_index], inplace=True)
                    if not ewh_ponta_day.empty:
                        ponta_starting_index = ewh_ponta_day.index[0]

            # Check for cheia profiles
            if sun_surplus_day[time_step] > 0 \
                    and not ewh_cheia_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_cheia_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available):
                    ewh_available = int(ewh_cheia_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[cheia_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_cheia_day.drop([cheia_starting_index], inplace=True)
                if not ewh_cheia_day.empty:
                    cheia_starting_index = ewh_cheia_day.index[0]

            # Check for vazio profiles
            if sun_surplus_day[time_step] > 0 \
                    and not ewh_vazio_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_vazio_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available):
                    ewh_available = int(ewh_vazio_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[vazio_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_vazio_day.drop([vazio_starting_index], inplace=True)
                if not ewh_vazio_day.empty:
                    vazio_starting_index = ewh_vazio_day.index[0]

            # Check for supervazio profiles
            if sun_surplus_day[time_step] > 0 \
                    and not ewh_supervazio_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_supervazio_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available):
                    ewh_available = int(ewh_supervazio_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[supervazio_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_supervazio_day.drop([supervazio_starting_index], inplace=True)
                if not ewh_supervazio_day.empty:
                    supervazio_starting_index = ewh_supervazio_day.index[0]

            # change the available status
            ewh_status_day[time_step] += sum(new_ewh_status_day[time_step, :])
        new_ewh_status[:, day, number_ewh] = ewh_status_day

    new_ewh_status_df = new_ewh_status[:, 0, number_ewh]
    for i in range(1, 28):
        new_ewh_status_df = np.append(arr=new_ewh_status_df, values=new_ewh_status[:, i, number_ewh])
    df['new ewh status'] = pd.Series(new_ewh_status_df)

    # recalculate flex vector
    df['new flex'] = df['new ewh status'].values * ewh_consumption_single
    # Get the cost for the scenario
    df['cost3'] = v_cost_period(df['demand'].values, df['new flex'].values,
                                df['pv production'].values, df["grid price"].values, n_set3)
    cost3[n_set3] = df['cost3'].sum() + n_set3 * 5 * LC

print(min(cost2))
print(min(cost3))
print(cost2.argmin())
print(cost3.argmin())
df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, cost3.argmin())
plt.plot(df['sun surplus'], 'r')
plt.plot(df['new flex'], '--')
plt.plot(df['flex'])
plt.grid()
plt.show()
