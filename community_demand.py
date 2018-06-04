import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Open LV profiles csv file
df = pd.read_excel('Dados de Fev.xlsx', 'Raw')
df.index = df['Referência Equipamento']

reference = df['Referência Equipamento'].unique()
total_id = len(reference)
date = df['Data da leitura'].unique()
total_date = len(date)

final_consumption = np.asarray([0 for x in range(672)])

# Check if profiles have at least 7 days of data in a row
for i in range(0, total_id):
    df1 = df.loc[df['Referência Equipamento'] == reference[i]]
    count = 0
    for j in range(0, total_date):
        df2 = df1.loc[df1['Data da leitura'] == date[j]]
        check_row = df2.shape[0]
        if check_row == 96:
            count += 1
        else:
            count = 0
        if count == 7:
            # Condition satisfied // create a dataframe with the 7 days in a row
            for k in range(1, 7):
                df3 = df1.loc[df1['Data da leitura'] == (date[j] - pd.Timedelta(days=k))]
                frames = [df3, df2]
                df2 = pd.concat(frames)

            # Order the dataframe from Monday to Sunday
            df2['Weekday'] = df2['Data da leitura'].dt.weekday
            df3 = df2.loc[df2['Weekday'] == 0]
            for z in range(1, 7):
                df4 = df2.loc[df2['Weekday'] == z]
                frames = [df3, df4]
                df3 = pd.concat(frames)

            # Sum the profile consumption to total final consumption
            final_consumption = final_consumption + df3['A+ (Total) kWh']
            final_consumption = np.asarray(final_consumption)       # in kWh!
            break

        # Drop values that doesn't have 7 days in a row from df to speed up the for cycle
        if (total_date - 1) == j:
            df.drop(labels=reference[i], inplace=True)

df_print = pd.DataFrame({
    'final consumption winter': final_consumption
})
times = pd.date_range('14/03/2016 00:15', periods=672, freq='15min')
df_print.index = times
df_print.to_csv('community_demand.csv', index_label='time')


# # Add EWH consumption to specific profiles
reference_check = df['Referência Equipamento'].unique()
total_id_check = len(df['Referência Equipamento'].unique())

# get the profiles with 6900 VA contracted
for i in range(total_id_check):
    df1 = df.loc[df['Referência Equipamento'] == reference_check[i]]
    contracted_power = np.asarray(df1['Potência Contratada (VA)'].head(1))
    if contracted_power != 6900:
        df.drop(labels=reference[i], inplace=True)

reference_6900 = df['Referência Equipamento'].unique()      # need to get just a certain amount
ewh_on = pd.Series(np.zeros(672), index=times)
ewh_consumption = pd.Series(np.zeros(672), index=times)

for i in range(len(reference_6900)):
    df1 = df.loc[df['Referência Equipamento'] == reference_6900[i]]
    count = 0
    for j in range(0, total_date):
        df2 = df1.loc[df1['Data da leitura'] == date[j]]
        check_row = df2.shape[0]
        if check_row == 96:
            count += 1
        else:
            count = 0
        if count == 7:
            for k in range(1, 7):
                df3 = df1.loc[df1['Data da leitura'] == (date[j] - pd.Timedelta(days=k))]
                frames = [df3, df2]
                df2 = pd.concat(frames)
            df2['Weekday'] = df2['Data da leitura'].dt.weekday
            df3 = df2.loc[df2['Weekday'] == 0]
            for z in range(1, 7):
                df4 = df2.loc[df2['Weekday'] == z]
                frames = [df3, df4]
                df3 = pd.concat(frames)
            df3.index = times
            for z in range(7):
                df4 = df3.loc[df3['Weekday'] == z]
                index_max = df4['A+ (Total) kWh'].idxmax()
                ewh_on[index_max] += 1          # maybe shift a bit the ewh consumption from the peak // need to make...
                # ...to make it on for 2 more consecutive periods

# EWH with 4.5 kW power rated power
rated_power = 4.5
ewh_consumption_single = rated_power * .25  # consumption per period in kWh

for clock in range(672):
    # if clock < 669:
        ewh_consumption[clock] = ewh_on[clock] * ewh_consumption_single

plt.plot(ewh_consumption)
plt.show()
