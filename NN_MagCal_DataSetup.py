# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:05:52 2018

@author: z3332903
"""

import numpy as np
import numpy.random
import random
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
import math
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import matlab.engine

#from buccy data, the measure values will need to be backed out from the cal'd data

buccyGains = np.array([0.84,0.79,0.81]) # [Gx,Gy,Gz]
buccyBiases = np.array([1363,-227,41])
biasFactor = 0.032
buccyBiases = buccyBiases*biasFactor
teslaFactor = 31.24e-09

cwd = os.getcwd()

# assign spreadsheet file name tol 'file'
file = '2018-08-03T14_34_40.636_ResponseData.xlsx'

# load spreadsheet
xl = pd.ExcelFile(file)

# load a sheet into a dataframe by name:df1
df1 = xl.parse('2018-08-03T14_34_40.636_Respons')

a = (df1['adcs_iBfield0'])

b = (df1['adcs_iBfield1'])

c = (df1['adcs_iBfield2'])

d = (df1['adcs_SpacecraftPos_ECI0'])

e = (df1['adcs_SpacecraftPos_ECI1'])

f = (df1['adcs_SpacecraftPos_ECI2'])

g = (df1['TIME_COLLECTED'])

dataCald = [[],[],[],[],[],[],[]]
data = [[],[],[],[],[],[],[],[]]
pos = []


for i in range(len(a)):
    if math.isnan(a[i]) == False:
        if a[i] != 0:
            dataCald[0].append(a[i])
            dataCald[1].append(b[i])
            dataCald[2].append(c[i])
            dataCald[3].append(d[i])
            dataCald[4].append(e[i])
            dataCald[5].append(f[i])
            dataCald[6].append(g[i])

#print (len(dataCald))         
         
for s in range(len(dataCald)):    
        for a in range(len(dataCald[s])):
            if s <=2:        
                cald = dataCald[s][a]
                magMeasured = (cald/buccyGains) - buccyBiases
                data[s].append(magMeasured)
            if s ==6:
                data[s].append(Time(dataCald[s][a]))
            else:
                data[s].append(dataCald[s][a])
                
for s in range(len(data[3])):
    cartrep = coord.CartesianRepresentation(x=data[3][s],y=data[4][s],z=data[5][s],unit=u.km)
    gcrs = coord.GCRS(cartrep, obstime=data[6][s])
    itrs = gcrs.transform_to(coord.ITRS(obstime=data[6][s]))
    loc = coord.EarthLocation(itrs.x,itrs.y,itrs.z)
    lon = loc.lon.to_value()
    lat = loc.lat.to_value()
    h = loc.height.to_value()
    pos.append([lat,lon,h])
    #pos.append([loc.lat,loc.lon,loc.height])
    

print (pos[0])


print (geomag.mag_heading(pos[0][2],pos[0][0],pos[0][1]))


    
    

              
 
       

