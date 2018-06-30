import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location

# Location data for Martim Longo
latitude = 37.442
longitude = -7.771
tz = 'UTC'
altitude = 249
location = Location(latitude, longitude, tz=tz, altitude=altitude)

# Weather data from Meteonorm, 1 min resolution
df1 = pd.read_excel('Marim_Longo-min.xlsx')
df1['times'] = pd.date_range('1/1/2016', periods=525600, freq='min')
df1.index = df1['times']
df1.index.name = 'Time'

# Get the portion of data required:
# 1 week of March starting from the 14th
# 1 week of June starting from the 6th
# 1 week of September starting from the 12th
# 1 week of December starting from the 5th
df_march = df1.loc['20160314':'20160320']
df_june = df1.loc['20160606':'20160612']
df_september = df1.loc['20160912':'20160918']
df_december = df1.loc['20161205':'20161217']
frame = [df_march, df_june, df_september, df_december]
df = pd.concat(frame)

times = df['times']
d_weather = {'temp_air': df['Ta'],
             'wind_speed': df['FF'],
             'ghi': df['G_Gh'],
             'dni': df['G_Bn'],
             'dhi': df['G_Dh']}
weather_data = pd.DataFrame(d_weather)

# PV data
pv_db = pvlib.pvsystem.retrieve_sam('SandiaMod')
inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
# print(list(pv_db))
# print(list(inverter_db))
pv_data = pv_db['SolarWorld_Sunmodule_250_Poly__2013_']  # Hanwha_HSL60P6_PA_4_250T__2013_
inverter = inverter_db['SMA_America__SB5000TL_US_22__240V__240V__CEC_2013_']  # 'iPower__SHO_5_2__240V__240V__CEC_2018_'

# Production
system = PVSystem(surface_tilt=37, surface_azimuth=180, albedo=0.2,
                  surface_type=None, module_parameters=pv_data,
                  modules_per_string=10, strings_per_inverter=2,
                  inverter_parameters=inverter,
                  racking_model='open_rack_cell_glassback')
mc = ModelChain(system, location, orientation_strategy=None,
                clearsky_model='ineichen', transposition_model='perez',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                dc_model='sapm', ac_model=None, aoi_model='physical',
                spectral_model='no_loss', temp_model='sapm',
                losses_model='no_loss', name=None)

mc.run_model(times, None, weather_data)
power_ac = pd.Series(mc.ac)
power_ac.fillna(value=0, inplace=True)

# Summing up the energy produced each 15 min
energy_produced_min = power_ac / 60e3   # energy in kWh
energy_produced_15min = energy_produced_min.resample('15min').sum()

df_print = pd.DataFrame({
    'PV production': energy_produced_15min
})

new_index = pd.date_range('2016/03/14 00:15', periods=26784, freq='15min')
df_print.index = new_index
df_print.to_csv('pv_production_1.csv', index_label='time')

# print(energy_produced_15min.sum())
# print(energy_produced_min.sum())
# plt.plot(energy_produced_15min)
# plt.grid()
# plt.show()
