import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location

# Location data for Martim Longo
latitude = -33.45
longitude = -70.55
tz = 'UTC'
altitude = 570
location = Location(latitude, longitude, tz=tz, altitude=altitude)

# Weather data from Meteonorm, 1 min resolution
df = pd.read_csv('dati.csv', header=1)
df['times'] = pd.date_range('1/1/2016', periods=8760, freq='H')
df.index = df['times']
times = df['times'].values

d_weather = {'temp_air': df['Ta'],
             'wind_speed': df['FF'],
             'ghi': df['G_Gh'].values,
             'dni': df['G_Bn'].values,
             'dhi': df['G_Dh'].values}
weather_data = pd.DataFrame(d_weather)

# PV data
pv_db = pvlib.pvsystem.retrieve_sam('SandiaMod')
inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
# print(list(pv_db))
# print(list(inverter_db))
pv_data = pv_db['SolarWorld_Sunmodule_250_Poly__2013_']  # Hanwha_HSL60P6_PA_4_250T__2013_
inverter = inverter_db['SMA_America__SB5000TL_US_22__240V__240V__CEC_2013_']  # 'iPower__SHO_5_2__240V__240V__CEC_2018_'

# Production
system = PVSystem(surface_tilt=83, surface_azimuth=78, albedo=0.2,
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
energy_produced_min = power_ac    # energy in kWh
energy_produced_month = energy_produced_min.resample('M').sum()
print(energy_produced_month / 1e3 / 5)
print(energy_produced_month.values.sum() / 1e3)

df1 = pd.DataFrame(data=(energy_produced_month/1e3/5))
df1.to_csv('cile_prod.csv')


# print(energy_produced_min.sum())
# plt.plot(energy_produced_15min)
# plt.grid()
# plt.show()
