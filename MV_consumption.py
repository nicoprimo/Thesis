import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Open file with MV profiles consumption data
df = pd.read_excel('CCH.xlsx')
df['DataLectura'] = pd.to_datetime(df['DataLectura'], yearfirst=True)
df.index = df['DataLectura']
reference_id = df['CUPS'].unique()
df['weekday'] = df['DataLectura'].dt.weekday

# Sum up all the consumption for the reference weeks for each quarter
consumption_hour_march = np.asarray([0 for x in range(169)])
consumption_hour_june = np.asarray([0 for x in range(169)])
consumption_hour_september = np.asarray([0 for x in range(169)])
consumption_hour_december = np.asarray([0 for x in range(169)])

total_id = len(reference_id)
for i in range(total_id):
    df1 = df.loc[df['CUPS'] == reference_id[i]]
    df_week_march = df1.loc['20180221230000':'20180228']
    df_week_june = df1.loc['201706132300':'20170620']       # get May instead to avoid low consumption due holidays
    df_week_september = df1.loc['201709112300':'20170918']
    df_week_december = df1.loc['201711042300':'20171111']   # get November instead

    # Order from Monday to Sunday
    df2_march = df_week_march.loc[df_week_march['weekday'] == 0]
    df2_june = df_week_june.loc[df_week_june['weekday'] == 0]
    df2_september = df_week_september.loc[df_week_september['weekday'] == 0]
    df2_december = df_week_december.loc[df_week_december['weekday'] == 0]
    for j in range(1, 7):
        df3_march = df_week_march.loc[df_week_march['weekday'] == j]
        df3_june = df_week_june.loc[df_week_june['weekday'] == j]
        df3_september = df_week_september.loc[df_week_september['weekday'] == j]
        df3_december = df_week_december.loc[df_week_december['weekday'] == j]

        frame_march = [df2_march, df3_march]
        frame_june = [df2_june, df3_june]
        frame_september = [df2_september, df3_september]
        frame_december = [df2_december, df3_december]

        df2_march = pd.concat(frame_march)
        df2_june = pd.concat(frame_june)
        df2_september = pd.concat(frame_september)
        df2_december = pd.concat(frame_december)

    consumption_hour_march = df2_march['ActivaImport'] / 1000 + consumption_hour_march    # in kW
    consumption_hour_june = df2_june['ActivaImport'] / 1000 + consumption_hour_june
    consumption_hour_september = df2_september['ActivaImport'] / 1000 + consumption_hour_september
    consumption_hour_december = df2_december['ActivaImport'] / 1000 + consumption_hour_december

    consumption_hour_march = np.asarray(consumption_hour_march)
    consumption_hour_june = np.asarray(consumption_hour_june)
    consumption_hour_september = np.asarray(consumption_hour_september)
    consumption_hour_december = np.asarray(consumption_hour_december)
# Assume the reference week to be the selected one of the year 2016
new_index_march = pd.date_range('14/03/2016 0:15', periods=169, freq='H')
new_index_june = pd.date_range('06/06/2016 0:15', periods=169, freq='H')
new_index_september = pd.date_range('09/12/2016 0:15', periods=169, freq='H')
new_index_december = pd.date_range('12/05/2016 0:15', periods=169, freq='H')

consumption_hour_march = pd.Series(data=consumption_hour_march, index=new_index_march)
consumption_hour_june = pd.Series(data=consumption_hour_june, index=new_index_june)
consumption_hour_september = pd.Series(data=consumption_hour_september, index=new_index_september)
consumption_hour_december = pd.Series(data=consumption_hour_december, index=new_index_december)

# get the energy each 15 min
power_15min_march = consumption_hour_march.resample('15min').interpolate(method='linear')     # still in kW
power_15min_june = consumption_hour_june.resample('15min').interpolate(method='linear')
power_15min_september = consumption_hour_september.resample('15min').interpolate(method='linear')
power_15min_december = consumption_hour_december.resample('15min').interpolate(method='linear')

consumption_15min_march = power_15min_march * .25     # get the kWh
consumption_15min_june = power_15min_june * .25
consumption_15min_september = power_15min_september *.25
consumption_15min_december = power_15min_december * .25

consumption_15min_march.drop(consumption_15min_march.tail(1).index, inplace=True)
consumption_15min_june.drop(consumption_15min_june.tail(1).index, inplace=True)
consumption_15min_september.drop(consumption_15min_september.tail(1).index, inplace=True)
consumption_15min_december.drop(consumption_15min_december.tail(1).index, inplace=True)


df_print = pd.DataFrame({
    'final consumption march': consumption_15min_march,
    'final consumption june': consumption_15min_june,
    'final consumption september': consumption_15min_september,
    'final consumption december': consumption_15min_december
})
df_print.to_csv('MV_demand.csv', index_label='time')

plt.plot(consumption_hour_march)
plt.plot(consumption_hour_june)
plt.plot(consumption_15min_september)
plt.plot(consumption_15min_december)
plt.show()
