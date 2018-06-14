import pandas as pd
import numpy as np
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

# Read flex consumption - EWH rated power = 4.5 kW
flex = pd.concat(4 * [df_LV['flex winter']], ignore_index=True)  # get the consumption from flexible assets
ewh_status = pd.concat(4 * [df_LV['63']], ignore_index=True)  # get the status of clustered ewh
# Get the single EWH status
ewh_n_status = pd.concat(4 * [df_LV.drop(df_LV.columns[[0, 1, 65]], axis=1)], ignore_index=True)

# get only the desired profiles (higher than 0 less than 63)
number_ewh = 50
ewh_n_status = ewh_n_status.iloc[:, :number_ewh]

rated_power = 4.5
ewh_consumption_single = rated_power * .25  # consumption per period in kWh

# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'flex': flex,
                   'pv production': pv_production,
                   'ewh status': ewh_status,
                   'grid price': grid_price}, dtype=float)
df = pd.concat([df, ewh_n_status], axis=1)
df['period price'] = period_price

# Need to get real value for PV system
LC = 100 / 52 * 4  # Price in euro/kW of PV installed per 1 week (1 year = 52 weeks) * 4 weeks (reference ones)

feed_in_tariff = 0  # should be set as the average price in the stock market times 0.90


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

for n_set3 in range(100):  # See the price variations up to 100 PV systems
    # add flexibility part // Align as much as possible ewh status and sun surplus //
    df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, n_set3)

# Set up sun surplus with 57 PV systems
df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, 57)
new_ewh_status = np.zeros(shape=(96, 28, number_ewh))
for day in range(1):
    # Get the values of the day
    df_day = df[(day * 96):((day + 1) * 96)]

    period_price_day = df_day['period price'].values
    sun_surplus_day = df_day['sun surplus'].values

    ewh_status_day = df_day['ewh status'].values
    new_ewh_status_day = new_ewh_status[:, day, :]
    ewh_n_status_day = ewh_n_status[(day * 96):((day + 1) * 96)]
    # Get the profiles "ponta"
    ewh_ponta_day = ewh_n_status_day[df_day['period price'] == 'p'].where(ewh_n_status_day > 0)
    ewh_ponta_day.dropna(axis=1, how='all', inplace=True)
    ewh_ponta_day.dropna(axis=0, how='all', inplace=True)
    ponta_starting_index = ewh_ponta_day.index[0]

    # Get the profiles "cheia"
    ewh_cheia_day = ewh_n_status_day[df_day['period price'] == 'c'].where(ewh_n_status_day > 0)
    ewh_cheia_day.dropna(axis=1, how='all', inplace=True)
    ewh_cheia_day.dropna(axis=0, how='all', inplace=True)
    cheia_starting_index = ewh_cheia_day.index[0]

    # Get the profiles "vazio"
    ewh_vazio_day = ewh_n_status_day[df_day['period price'] == 'v'].where(ewh_n_status_day > 0)
    ewh_vazio_day.dropna(axis=1, how='all', inplace=True)
    ewh_vazio_day.dropna(axis=0, how='all', inplace=True)
    vazio_starting_index = ewh_vazio_day.index[0]

    # Get the profiles "super vazio"
    ewh_supervazio_day = ewh_n_status_day[df_day['period price'] == 'sv'].where(ewh_n_status_day > 0)
    ewh_supervazio_day.dropna(axis=1, how='all', inplace=True)
    ewh_supervazio_day.dropna(axis=0, how='all', inplace=True)
    supervazio_starting_index = ewh_supervazio_day.index[0]

    for time_step in range(96):
        # Fill up sun surplus
        if sun_surplus_day[time_step] > 0:
            # Check if there is any ponta profile before time_step
            if (time_step - ponta_starting_index) >= 0 and not ewh_ponta_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_ponta_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available - 1):
                    ewh_available = int(ewh_ponta_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[ponta_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_ponta_day.drop([ponta_starting_index], inplace=True)
                if not ewh_ponta_day.empty:
                    ponta_starting_index = ewh_ponta_day.index[0]

            # Check for cheia profiles
            if (time_step - cheia_starting_index) >= 0 and sun_surplus_day[time_step] > 0 \
                    and not ewh_cheia_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_cheia_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available - 1):
                    ewh_available = int(ewh_cheia_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[cheia_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_cheia_day.drop([cheia_starting_index], inplace=True)
                if not ewh_cheia_day.empty:
                    cheia_starting_index = ewh_cheia_day.index[0]

            # Check for vazio profiles
            if (time_step - vazio_starting_index) >= 0 and sun_surplus_day[time_step] > 0 \
                    and not ewh_vazio_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_vazio_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available - 1):
                    ewh_available = int(ewh_vazio_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[vazio_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_vazio_day.drop([vazio_starting_index], inplace=True)
                if not ewh_vazio_day.empty:
                    vazio_starting_index = ewh_vazio_day.index[0]

            # Check for supervazio profiles
            if (time_step - supervazio_starting_index) >= 0 and sun_surplus_day[time_step] > 0 \
                    and not ewh_supervazio_day.empty:
                # Get how many EWH profiles are available
                profile_available = ewh_supervazio_day.head(1).dropna(axis=1).shape[1]
                # Check if there is the need to shift all the profiles
                if profile_available > sun_surplus_day[time_step]:
                    profile_available = sun_surplus_day[time_step]
                # Shift the profiles available to the sun surplus
                for i in range(profile_available - 1):
                    ewh_available = int(ewh_supervazio_day.head(1).dropna(axis=1).columns[i])
                    new_ewh_status_day[time_step, ewh_available] += 1
                    new_ewh_status_day[supervazio_starting_index, ewh_available] -= 1
                # Update index of first values available and drop the 1st row of profiles used
                sun_surplus_day[time_step] -= profile_available
                ewh_supervazio_day.drop([supervazio_starting_index], inplace=True)
                if not ewh_supervazio_day.empty:
                    supervazio_starting_index = ewh_supervazio_day.index[0]
    # change the available status
    for time_step in range(96):
        ewh_status_day[time_step] += sum(new_ewh_status_day[time_step, :])



