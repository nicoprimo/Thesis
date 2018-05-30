import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Open file with MV profiles consumption data
df = pd.read_excel('CCH.xlsx')
df['DataLectura'] = pd.to_datetime(df['DataLectura'], yearfirst=True)
df.index = df['DataLectura']
reference_id = df['CUPS'].unique()
df['weekday'] = df['DataLectura'].dt.weekday

# Sum up all the consumption for the week that finish the 28th of February
consumption_hour = np.asarray([0 for x in range(169)])
total_id = len(reference_id)
for i in range(total_id):
    df1 = df.loc[df['CUPS'] == reference_id[i]]
    df_week = df1.loc['20180221230000':'20180228']
    # Order from Monday to Sunday
    df2 = df_week.loc[df_week['weekday'] == 0]
    for j in range(1, 7):
        df3 = df_week.loc[df_week['weekday'] == j]
        frame = [df2, df3]
        df2 = pd.concat(frame)

    consumption_hour = df2['ActivaImport'] / 1000 + consumption_hour    # in kW
    consumption_hour = np.asarray(consumption_hour)

new_index = pd.date_range('16/03/2016', periods=169, freq='H')
consumption_hour = pd.Series(data=consumption_hour, index=new_index)
consumption_15min = consumption_hour.resample('15min').interpolate(method='linear')     # still in kW
consumption_15min = consumption_15min * .25     # get the kWh
consumption_15min.drop(consumption_15min.tail(1).index, inplace=True)

df_print = pd.DataFrame({
    'final consumption': consumption_15min
})
df_print.to_csv('MV_demand.csv')
