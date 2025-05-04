# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only
#
# Note: This file is a special version of the gpslib.py module from the
# project GPS-SDR-Receiver. It is intended for use with the GPS-Tutorial.

import numpy as np
from scipy.fft import fft, ifft
import datetime
import math
try: 
    from pycode.cacodes import cacodes 
except ImportError:
    from cacodes import cacodes 

SAMPLE_RATE = 2048000 
WEEK_IN_SEC = 604800
GPS_C  = 2.99792458e8           # as defined for GPS
GPS_PI = 3.1415926535898        # dito
OMEGA_EARTH = 7.292115147e-5    # rad/sec

# ----------- Sets of quantities in subframes or from calculations -----------------

# just for information - not used in code
# ST (arrival time of subframe) and DEL (cacode phase shift) both in units of 1/samplerate, added at decoding of data
subFrame1 = {'ID','tow','weekNum','satAcc','satHealth','Tgd','IODC','Toc','af2','af1','af0','ST'} 
subFrame2 = {'ID','tow','Crs','deltaN','M0','Cuc','IODE2','e','Cus','sqrtA','Toe','ST'} 
subFrame3 = {'ID','tow','Cic','omegaBig','Cis','i0', 'IODE3','Crc','omegaSmall','omegaDot','IDOT','ST'} 
subFrame4 = {'ID','tow','ST'}  # mostly almanach data
subFrame5 = {'ID','tow','ST'}  # mostly almanach data 
# data in SF 4 for calculation of time delay caused by ionosphere is not yet used

# used in Class SatData() to extract needed information from subframes
ephemSF1  = {'weekNum','Tgd','Toc','af2','af1','af0','IODC','satAcc'} 
ephemSF2  = {'Crs','deltaN','M0','Cuc','e','Cus','sqrtA','Toe','IODE2'} 
ephemSF3  = {'Cic','omegaBig','Cis','i0','Crc','omegaSmall','omegaDot','IDOT','IODE3'} 

# Calculated quantities in class SatPos() - not used in code
orbCalc  = {'Omega_k','M_k','E_k','nu_k','Phi_k','d_ik','i_k','d_uk','u_k','d_rk','r_k','op_xk','op_yk'}       
timeCalc = {'tsv','dtsv','gpst','tk','dtr'}
                                    
# Needed representation of quantities  -  may be helpful for saving in text files (not yet used)
boolVal  = {'SWP'}
intVal   = {'SAT','ID','tow','ST','weekNum','satAcc','satHealth','IODC','IODE2','IODE3','Toc','Toe'}
floatVal = {'height','lat','lon','AMP','CRM','FRQ','Tgd','af2','af1','af0','Crs','deltaN','M0','Cuc','e','Cus','sqrtA','Cic',\
            'omegaBig','Cis','i0','Crc','omegaSmall','omegaDot','IDOT','Omega_k','M_k','E_k','nu_k',\
            'Phi_k','d_ik','i_k','d_uk','u_k','d_rk','r_k','op_xk','op_yk','tsv','dtsv','gpst','tk','dtr'}
string = {'EPH'}


# -------- floating point types used in arrays ---------
# 32-bit types overall ~1.4 times faster than 64-bit
# (factor for single multiplication ~2)

MY_FLOAT = np.float32
MY_COMPLEX = np.complex64

# ------ original GPS cacodes -----------------

def origGPSCacode(satNo):
    y = cacodes[satNo]
    y = np.asarray(y,dtype=MY_FLOAT)
    x = np.arange(len(y))
    x = np.asarray(x,dtype=MY_FLOAT)
    return x, y

# ------ doubled GPS cacodes -------------------

def doubledGPSCacode(satNo):
    code = cacodes[satNo]
    y = []
    for i in range(len(code)):
        y += [code[i],code[i]]
    y = np.asarray(y,dtype=MY_FLOAT)
    x = np.arange(len(y))
    x = np.asarray(x,dtype=MY_FLOAT)
    return x, y

# ------ Interpolation of cacode to 2048 points (CODE_SAMPLES) ----

def GPSCacode(satNo,codeSamples):
    x,y = doubledGPSCacode(satNo)
    xp = np.linspace(x[0],x[-1],codeSamples,endpoint=True,dtype=MY_FLOAT)
    yp = np.interp(xp,x,y)       
    return yp

# ------------------------------------------------

def GPSCacodeRep(satNo,codeSamples,NCopies,delay):
    y0 = GPSCacode(satNo,codeSamples)
    y1 = y0
    for _ in range(NCopies-1):
        y1 = np.append(y1,y0)
    y1 = np.roll(y1,delay)
    return y1
                        

# ----------- Class Subframe ---------------------------------------------------
# Input for the function Extract is a stream of bits (int8, 1 and 0) representing 
# a subframe of length 300 beginning with the preamble (normal or inverted). Output 
# is a status word. If valid the decoded subframe parameters are saved in corresponding 
# variables of the object instance.
   
class Subframe():
    errlst = ['no error',              
              'wrong length of subframe',
              'preamble error',
              'parity error',
              'no valid ID',
              'empty']
    noErr = 0
    lengthErr = 1          
    preambleErr = 2
    parityErr = 3
    idErr = 4
    noData = 5
    preamble = np.array([1,0,0,0,1,0,1,1],dtype=np.int8)
    rollover = 2                                     # current rollover of weekNum

    def __init__(self):
        self._words = np.zeros((10,30),dtype=np.int8)
        self._status = self.noData
        # all subframes
        self._ID = 0             # subframe ID 1 to 5 
        self._tow = 0            # time of week count
        # subframe 1
        self._weekNum = 0        # week number
        self._satAcc = 0         # satellite accuracy
        self._satHealth = 0      # satellite health
        self._Tgd = 0            # in s, correction term for group delay differential
        self._IODC = 0           # Issue of Data, for detection of changes in parameters, see IODE2, IODE3
        self._Toc = 0            # clock data reference time - for calculation of code phase offset 
        self._af2 = 0            # polynomial coefficient - for calculation of code phase offset 
        self._af1 = 0            # polynomial coefficient - for calculation of code phase offset 
        self._af0 = 0            # polynomial coefficient - for calculation of code phase offset 
        # subframe 2
        self._IODE2 = 0          # Issue of Data (ephemeris), see IODC, IODE3
        self._Crs = 0            # for correction of orbit radius
        self._deltaN = 0         # mean motion difference from computed value
        self._M0 = 0             # mean anomaly at reference time
        self._Cuc = 0            # for correction of argument of latitude
        self._Cus = 0            # for correction of argument of latitude
        self._e = 0              # eccentricity
        self._sqrtA = 0          # square root of the semi-major axis
        self._Toe = 0            # reference time ephemeris (or epoch time)
        # subframe 3
        self._Cic = 0            # for correction of angle of inclination
        self._Cis = 0            # for correction of angle of inclination
        self._omegaBig = 0       # longitude of ascending node of orbit plane at weekly epoch
        self._i0 = 0             # inclination angle at reference time
        self._Crc = 0            # for correction of orbit radius
        self._omegaSmall = 0     # argument of perigree ('erdnächster Ort')
        self._omegaDot = 0       # rate of right ascension
        self._IDOT = 0           # rate of inclination angle
        self._IODE3 = 0          # Issue of Data (ephemeris), see IODE2, IODC

    @property 
    def word(self,value):        
        return self._words[value,:]

    @property 
    def ID(self):        
        return self._ID
    
    @property 
    def tow(self):        
        return self._tow
    
    @property 
    def status(self):        
        return self._status
    
    @property 
    def errmsg(self):                
        return self.errlst[self._status]
    
    @property 
    def weekNum(self):
        return self._weekNum

    @property 
    def satAcc(self):
        return self._satAcc

    @property 
    def satHealth(self):
        return self._satHealth

    @property 
    def Tgd(self):
        return self._Tgd

    @property 
    def IODC(self):
        return self._IODC

    @property 
    def Toc(self):
        return self._Toc

    @property 
    def af2(self):
        return self._af2

    @property 
    def af1(self):
        return self._af1

    @property 
    def af0(self):
        return self._af0
    @property 
    def IODE2(self):
        return self._IODE2

    @property 
    def Crs(self):      
        return self._Crs

    @property 
    def deltaN(self):
        return self._deltaN               

    @property 
    def M0(self):
        return self._M0

    @property 
    def Cuc(self):
        return self._Cuc

    @property 
    def e(self):
        return self._e

    @property 
    def Cus(self):
        return self._Cus

    @property 
    def sqrtA(self):
        return self._sqrtA

    @property 
    def Toe(self):
        return self._Toe

    @property 
    def Cic(self):
        return self._Cic

    @property 
    def omegaBig(self):
        return self._omegaBig

    @property 
    def Cis(self):
        return self._Cis

    @property 
    def i0(self):
        return self._i0

    @property 
    def Crc(self):
        return self._Crc

    @property 
    def omegaSmall(self):
        return self._omegaSmall

    @property 
    def omegaDot(self):
        return self._omegaDot

    @property 
    def IDOT(self):
        return self._IDOT

    @property 
    def IODE3(self):
        return self._IODE3   

    def Extract(self,subframeData):
        if len(subframeData) != 300:
            self._status = self.lengthErr
        else:   
            data = np.copy(subframeData)
            preOk = (data[:8] == self.preamble).all()
            if not preOk:
                data = 1 - data
                preOk = (data[:8] == self.preamble).all()
            if not preOk:
                self._status = self.preambleErr
            else:
                self._words = np.reshape(data,(10,30))
                if self.CheckParity() > 0:
                    self._status = self.parityErr        
                else:
                    self._tow = self.BinToInt(self._words[1,:17])
                    self._ID = self.BinToInt(self._words[1,19:22])
                    if self._ID < 1 or self._ID > 5:
                        self._status = self.idErr
                    else:
                        if self._ID == 1:
                            self.getDataSub1()
                        elif self._ID == 2:
                            self.getDataSub2()
                        elif self._ID == 3:
                            self.getDataSub3()
    #                    elif self._ID == 4:
    #                        self.getDataSub4()
    #                    elif self._ID == 5:
    #                        self.getDataSub5()                   
                        self._status = self.noErr                    
        return self._status
    
    def getDataSub1(self):
        self._weekNum = self.BinToInt(self._words[2,:10])
        self._satAcc = self.BinToInt(self._words[2,12:16]) 
        self._satHealth = self.BinToInt(self._words[2,16:22]) 
        self._IODC = self.BinToInt(np.append(self._words[2,22:24],self._words[7,:8])) 
        self._Tgd = self.BinToInt(self._words[6,16:24],signed=True)*2**(-31)     # in s
        self._Toc = self.BinToInt(self._words[7,8:24])*16                        # in s; SV clock correction reference time
        self._af2 = self.BinToInt(self._words[8,0:8],signed=True)*2.0**(-55)     # s/s^2 ; SV clock drift rate correction 
        self._af1 = self.BinToInt(self._words[8,8:24],signed=True)*2.0**(-43)    # s/s; SV clock drift correction
        self._af0 = self.BinToInt(self._words[9,0:22],signed=True)*2.0**(-31)    # s; SV clock bias correction
        
    def getDataSub2(self):
        self._IODE2 = self.BinToInt(self._words[2,0:8])
        self._Crs = self.BinToInt(self._words[2,8:24],signed=True)*2.0**(-5)     # m 
        self._deltaN = self.BinToInt(self._words[3,0:16],signed=True)*2.0**(-43)*GPS_PI # rad/s
        self._M0 = self.BinToInt(np.append(self._words[3,16:24],self._words[4,0:24]),signed=True)*2.0**(-31)*GPS_PI # rad
        self._Cuc = self.BinToInt(self._words[5,0:16],signed=True)*2.0**(-29)    # rad
        self._e = self.BinToInt(np.append(self._words[5,16:24],self._words[6,0:24]))*2**(-33) # dimensionless
        self._Cus = self.BinToInt(self._words[7,0:16],signed=True)*2.0**(-29)    # rad 
        self._sqrtA = self.BinToInt(np.append(self._words[7,16:24],self._words[8,0:24]))*2.0**(-19) # m^(1/2)
        self._Toe = self.BinToInt(self._words[9,0:16])*16                        # s
        
    def getDataSub3(self):
        self._Cic = self.BinToInt(self._words[2,0:16],signed=True)*2.0**(-29)    # rad
        self._omegaBig = self.BinToInt(np.append(self._words[2,16:24],self._words[3,0:24]),signed=True)*2.0**(-31)*GPS_PI # rad
        self._Cis = self.BinToInt(self._words[4,0:16],signed=True)*2.0**(-29)    # rad
        self._i0 = self.BinToInt(np.append(self._words[4,16:24],self._words[5,0:24]),signed=True)*2.0**(-31)*GPS_PI # rad
        self._Crc = self.BinToInt(self._words[6,0:16],signed=True)*2.0**(-5)     # m
        self._omegaSmall = self.BinToInt(np.append(self._words[6,16:24],self._words[7,0:24]),signed=True)*2.0**(-31)*GPS_PI # rad
        self._omegaDot = self.BinToInt(self._words[8,0:24],signed=True)*2.0**(-43)*GPS_PI # rad/s
        self._IDOT = self.BinToInt(self._words[9,8:22],signed=True)*2.0**(-43)*GPS_PI   # rad/s 
        self._IODE3 = self.BinToInt(self._words[9,0:8])

#    def getDataSub4(self):
#        return 0
    
#    def getDataSub5(self):
#        return 0

    def CheckParity(self):
        res = 0
        i = 1
        while res == 0 and i < 10:
            DS29 = self._words[i-1,28]
            DS30 = self._words[i-1,29]
            d = self._words[i,:24]
            if DS30 == 1:
                d = 1 - d
                self._words[i,:24] = d
            D25 = DS29^d[0]^d[1]^d[2]^d[4]^d[5]^d[ 9]^d[10]^d[11]^d[12]^d[13]^d[16]^d[17]^d[19]^d[22]
            D26 = DS30^d[1]^d[2]^d[3]^d[5]^d[6]^d[10]^d[11]^d[12]^d[13]^d[14]^d[17]^d[18]^d[20]^d[23]
            D27 = DS29^d[0]^d[2]^d[3]^d[4]^d[6]^d[ 7]^d[11]^d[12]^d[13]^d[14]^d[15]^d[18]^d[19]^d[21]
            D28 = DS30^d[1]^d[3]^d[4]^d[5]^d[7]^d[ 8]^d[12]^d[13]^d[14]^d[15]^d[16]^d[19]^d[20]^d[22]
            D29 = DS30^d[0]^d[2]^d[4]^d[5]^d[6]^d[ 8]^d[ 9]^d[13]^d[14]^d[15]^d[16]^d[17]^d[20]^d[21]^d[23]
            D30 = DS29^d[2]^d[4]^d[5]^d[7]^d[8]^d[ 9]^d[10]^d[12]^d[14]^d[18]^d[21]^d[22]^d[23]
            if (np.array([D25,D26,D27,D28,D29,D30],dtype=np.int8) != self._words[i,24:]).any():
                res = i
            i += 1
        return res
                
        
    def BinToInt(self,bits,signed=False):        # signed as two's complement
        z = 0
        f = 1
        neg = signed and bits[0]==1
        if neg:
            bits = 1-bits
        for b in reversed(bits):
            z += int(b)*f
            f *= 2
        if neg:
            z = -(z+1)
        return z
    
    
# ---------- Class SatPos(): calculates orbit parameter and position at a given time of a satellite ----------
# 
# Result: (X,Y,Z), orbit parameters and corrections of satellite time to be used for positioning
# Based on formulas in "Global Positioning System Standard Positioning Signal Specification", 2nd Edition (1995)

class SatPos():
    
    muE = 3.986005E+14                   # m^3/s^2; WGS-84 value of the Earth's universal gravitational parameter
    OmE = 7.2921151467E-5                # rad/sec; WGS-84 value of the Earth's rotation rate (angular velocity)
    F   = -4.44280763310E-10             # s/sqrt(m);  = -2 sqrt(mue)/c^2 
    rollover = 2                         # for GPS time, next rollover is on Sun, 21.11.2038 00:00 (+leapseconds for UTC)
    
    def __init__(self):
        self.dctOrb = {}
        self.dctTime = {}

    # Following quantities are calculated here:
    #
    # A     :   Semi-major axis
    # tk    :   Time from ephemeris reference epoch, 
    #           t is GPS system time at time of transmission, gps_toe is the epoc time
    # Mk    :   Corrected mean anomaly
    # Ek    :   Eccentric anomaly
    # nuk   :   True anomaly  
    # Phik  :   Argument of latitude
    # duk   :   Argument of latitude correction
    # rk    :   Corrected radius
    # drk   :   Further radius correction 
    # ik    :   Corrected inclination
    # dik   :   Further correction to inclination
    # uk    :   Corrected argument of latitude
    # Omegak:   Corrected longitude of ascending node        
    # opxk  :   x position in orbital plane
    # opyk  :   y position in orbital plane
    # xk    :   x position in Earth-Centered, Earth-Fixed coordinates
    # yk    :   x position in Earth-Centered, Earth-Fixed coordinates
    # zk    :   z position in Earth-Centered, Earth-Fixed coordinates
    #
    # Below, variables 'gps_...' are the corresponding quantities given in the subframes    
    
    def CrossTime(self,t):                  # account for beginning or end of week crossovers (p. 40)
        halfWeek = WEEK_IN_SEC // 2         # in seconds
        while t > halfWeek:
            t -= 2*halfWeek
        while t < -halfWeek:
            t += 2*halfWeek
        return t        
    
    def tsv(self,gps_tow):
        # nominal time of transmission, ideal GPS time - also called "effective SV PRN code phase time"  
        # To get the actual gps time, tsv has to be corrected using dtsv, t = tsv - dtsv
        return (gps_tow-1)*6

    def dtsv(self,t_sv,gps_af0,gps_af1,gps_af2,gps_toc,gps_tgd,dtr=0):                                             # dtr ADDED
        return gps_af0+gps_af1*self.CrossTime(t_sv-gps_toc)+gps_af2*self.CrossTime(t_sv-gps_toc)**2+dtr-gps_tgd    # dtr ADDED
        
    def gpst(self,t_sv,dt_sv):
    # exact GPS system time at time of transmission; here only from beginning of week epoch (Sunday, 0:00) 
        return t_sv - dt_sv
    
    def tk(self,gps_t,gps_toe):   # Time from ephemeris reference epoch           
        # difference between gps time and the epoch time toe
        # must account for beginning or end of week crossovers. 
        t_k = self.CrossTime(gps_t - gps_toe)
        return t_k
            
    def A(self,gps_sqrtA):                    # Semi-major axis
        return gps_sqrtA**2
            
    def Mk(self,t_k,gps_sqrtA,gps_deltaN,gps_M0):                  # Corrected mean anomaly
        return gps_M0 + (np.sqrt(self.muE)/gps_sqrtA**3+gps_deltaN)*t_k
    
    # Kepler equation: M=E-e*sin(E) with M mean anomaly and E eccentric anomaly
    def Ek(self,M_k,gps_e,itMax=10,eps=1.0E-12):                    # rad        
        q0 = 0
        E_k = M_k        
        it = 0
        while abs(E_k-q0) > eps and it < itMax:
            q0 = E_k
            E_k = q0-(q0-gps_e*np.sin(q0)-M_k)/(1-gps_e*np.cos(q0))            
            it += 1            
        return E_k
    
    def nuk(self,E_k,gps_e):                   # True anomaly  
        return np.arctan2(np.sqrt(1-gps_e**2)*np.sin(E_k),(np.cos(E_k)-gps_e))   
        # alternative, if not stable: Ek+2*np.arctan(beta*np.sin(Ek)/(1-beta*np.cos(Ek))) - Wikipedia
            
    def Phik(self,nu_k,gps_omegaSmall):             # Argument of latitude
        return nu_k + gps_omegaSmall
        
    def duk(self,Phi_k,gps_cus,gps_cuc):           # Argument of latitude correction
        return gps_cus*np.sin(2*Phi_k)+gps_cuc*np.cos(2*Phi_k)

    def drk(self,Phi_k,gps_crc,gps_crs):           # Corrected radius
        return gps_crc*np.cos(2*Phi_k)+gps_crs*np.sin(2*Phi_k)

    def dik(self,Phi_k,gps_cic,gps_cis):           # Correction to inclination
        return gps_cic*np.cos(2*Phi_k)+gps_cis*np.sin(2*Phi_k)
    
    def uk(self,Phi_k,d_uk):                        # Corrected argument of latitude
        return Phi_k + d_uk
    
    def rk(self,E_k,d_rk,gps_sqrtA,gps_e):                  # Corrected radius
        return gps_sqrtA**2 * (1-gps_e*np.cos(E_k))+d_rk
    
    def ik(self,t_k,d_ik,gps_i0,gps_idot):          # Corrected inclination
        return gps_i0 + d_ik + gps_idot*t_k
    
    def opxk(self,r_k,u_k):                         # x position in orbital plane
        return r_k*np.cos(u_k)

    def opyk(self,r_k,u_k):                         # y position in orbital plane
        return r_k*np.sin(u_k)

    def Omegak(self,t_k,gps_omegaBig,gps_omegaDot,gps_toe):   # Corrected longitude of ascending node        
        return gps_omegaBig + (gps_omegaDot-self.OmE)*t_k - self.OmE*gps_toe
    
    def xk(self,op_xk,op_yk,i_k,Omega_k):            # x positiion in Earth-Centered, Earth-Fixed coordinates
        return op_xk*np.cos(Omega_k) - op_yk*np.cos(i_k)*np.sin(Omega_k)

    def yk(self,op_xk,op_yk,i_k,Omega_k):            # x positiion in Earth-Centered, Earth-Fixed coordinates
        return op_xk*np.sin(Omega_k) + op_yk*np.cos(i_k)*np.cos(Omega_k)
    
    def zk(self,op_yk,i_k):                         # z position in Earth-Centered, Earth-Fixed coordinates
        return op_yk*np.sin(i_k)
    
    # --------- Calculate here the relevant coordinates for parameters given in subframes ---------
    # eph is a dictionary for the orbital parameters from the subframes 1,2,3

    def gpsUTCStr(self,gps_tow,eph):
        leaptime = 18          # current difference of GPS and UTC time in sec
        gps_t = self.gpsTime(gps_tow,eph)
        d = datetime.datetime(1980,1,6) + datetime.timedelta(seconds=round(gps_t-leaptime))

        return d.strftime('%a, %d.%m.%Y %H:%M:%S UTC')   # or with ms: ('%a, %d.%m.%Y %H:%M:%S.%f UTC')[:-7]        
        
    def gpsTime(self,gps_tow,eph):  
        # according to definition on p. 40, eqn. 1-2     
        t_sv = self.tsv(gps_tow)
        dt_sv = self.dtsv(t_sv,eph['af0'],eph['af1'],eph['af2'],eph['Toc'],eph['Tgd'])
        gps_t = (eph['weekNum']+self.rollover*1024)*WEEK_IN_SEC + self.gpst(t_sv,dt_sv)                
        return gps_t
        

    def ecefCoord(self,gps_tow,eph,DT=0,relCorr=True):      

        itRelCorr = 2 if relCorr else 1
        t_sv = self.tsv(gps_tow) + DT   # start time of subframe with given tow plus DT in sec
        t_gd = eph['Tgd']
        t_oc = eph['Toc']
        t_oe = eph['Toe']        
        dtr = 0                                                     
        for it in range(itRelCorr):                                          
            dt_sv = self.dtsv(t_sv,eph['af0'],eph['af1'],eph['af2'],t_oc,t_gd,dtr=dtr)     
            gps_t = self.gpst(t_sv,dt_sv) 
            t_k = self.tk(gps_t,t_oe)
            M_k = self.Mk(t_k,eph['sqrtA'],eph['deltaN'],eph['M0'])
            E_k = self.Ek(M_k,eph['e'])
            if itRelCorr and it==0:
                dtr = self.F*eph['e']*eph['sqrtA']*np.sin(E_k)       

        nu_k  = self.nuk(E_k,eph['e'])
        Phi_k = self.Phik(nu_k,eph['omegaSmall'])
        d_ik  = self.dik(Phi_k,eph['Cic'],eph['Cis'])
        i_k   = self.ik(t_k,d_ik,eph['i0'],eph['IDOT'])        
        d_uk = self.duk(Phi_k,eph['Cus'],eph['Cuc'])
        u_k = self.uk(Phi_k,d_uk)
        d_rk = self.drk(Phi_k,eph['Crc'],eph['Crs'])
        r_k = self.rk(E_k,d_rk,eph['sqrtA'],eph['e'])
        op_xk = self.opxk(r_k,u_k)
        op_yk = self.opyk(r_k,u_k)    
        
        Omega_k = self.Omegak(t_k,eph['omegaBig'],eph['omegaDot'],eph['Toe'])
        X = self.xk(op_xk,op_yk,i_k,Omega_k)
        Y = self.yk(op_xk,op_yk,i_k,Omega_k)
        Z = self.zk(op_yk,i_k)
        
        self.dctOrb = {'Omega_k': Omega_k,'M_k': M_k,'E_k': E_k,'nu_k': nu_k,'Phi_k': Phi_k,'d_ik': d_ik,'i_k': i_k, \
                       'd_uk': d_uk,'u_k': u_k,'d_rk': d_rk,'r_k': r_k,'op_xk': op_xk,'op_yk': op_yk}
        
        self.dctTime = {'tsv': t_sv,      # code phase time (for an ideal sender: GPS time)
                        'dtsv': dt_sv,    # correction to tsv (due to clock inaccuracies and propagation delay to antenna)
                        'dtr': dtr,       # relativistic correction
                        'gpst': gps_t,    # GPS time (only since last week epoch) 
                        'tk': t_k}        # for orbit calculation; time from ephemeris reference epoch 
                        # in frames:
                        #'Tgd': t_gd,      # group time delay, another parameter for dtsv
                        #'Toe': t_oe,      # epoch time or reference time ephemerides ("offset eph") 
                        #'Toc': t_oc,      # parameter for "offset code phase" dtsv
                                           # time since ephemeris reference epoch toc for calculating satellite orbit 
        
        return X,Y,Z,dt_sv            
    
# -------------------------- Class SatData() ------------------------------------------------
# Subframes from a satellite are passed to a SatData() instance to build ephemeris and a time 
# table of (tow,ST). A complete ephemeris table from subframes 1 to 3 is built by reading many 
# subframes. The validity of the subframes are checked and a possible change of ephemeris data 
# in the stream  of subframes is monitored. The time data tow and ST from all subframes 1-5 are 
# saved in a table.

class SatData():
    errlst = ['no error',      
              'not yet ready',
              'new ephemerides',
              'flawed frame',
              'not healthy']
    noErr = 0
    notReady = 1
    newEphem = 2
    flawedFrame = 3
    healthErr = 4
    
    def __init__(self,satNo,ephemeris=None):
        self._satNo = satNo
        self._status = 0
        self._ephemData = {}
        self._timeData = []
        self._ephemOk = False
        self._SF1 = False
        self._SF2 = False
        self._SF3 = False
        self._SFLst = []
        self._IODC  = -1
        self._IODE2 = -1
        self._IODE3 = -1
        self._Health = -1
        self._Accuracy = -1    # currently not used
        self._lastIODC = -1
        self._lastTow = -1
        self._lastST = -1        
        self.EPHEM_LOADED = (ephemeris is not None)
        if self.EPHEM_LOADED:
            self.loadEphem(ephemeris)
        
    @property 
    def errmsg(self):                
        return self.errlst[self._status]
    
    @property 
    def status(self):        
        return self._status
   
    @property 
    def ephemOk(self):        
        return self._ephemOk

    @property 
    def ephemData(self):        
        return self._ephemData

    @property 
    def timeData(self):        
        return self._timeData   
    
    @property 
    def subFrameLst(self):        
        return self._SFLst       
    
    @property 
    def IODC(self):        
        return self._IODC     
    
    def loadEphem(self,eph):
        self._ephemData = eph.copy()
        self._ephemData['SAT'] = self._satNo
        self._ephemOk = True
        self._SF1 = True
        self._SF2 = True
        self._SF3 = True
        self._IODC  = eph['IODC']
        self._IODE2 = eph['IODE2']
        self._IODE3 = eph['IODE3']
        self._Accuracy = eph['satAcc']
        self._Health = 0

        self._lastIODC = self._IODC        
    
    
    def framesValid(self,subframe):                
        maxShift = 6.5                              # max shift of cacode is ~3.15 chips in a second
                                                    # thus 6.3 in units of sampleTime (0.5µs)
        status = self.noErr
        iodc = -1
        if subframe['ID'] == 1: 
            iodc = subframe['IODC'] & 255
            self._Health = subframe['satHealth']
            if self._Health != 0:
                status = self.healthErr
        elif subframe['ID'] == 2: 
            iodc = subframe['IODE2']
        elif subframe['ID'] == 3: 
            iodc = subframe['IODE3']                    

        if status == self.noErr and iodc > -1:            
            if self._lastIODC > -1 and iodc != self._lastIODC:
                status = self.newEphem
            self._lastIODC = iodc

        self._lastTow = subframe['tow']
        self._lastST = subframe['ST']
        
        return status

        
    def readSubframe(self,subframe,saveSubframe = False):
        if saveSubframe:
            self._SFLst.append(subframe)                            
        self._status = self.framesValid(subframe)           # checks also iodc
        
        if self._status == self.noErr:                      # thus no new ephem from ID 1..3                  
            if not self._ephemOk:
                if subframe['ID'] == 1 and not self._SF1:
                    for key in ephemSF1:
                        self._ephemData[key]=subframe[key]
                    self._IODC = subframe['IODC']
                    self._Accuracy = subframe['satAcc']
                    self._SF1=True
                elif subframe['ID'] == 2 and not self._SF2:
                    for key in ephemSF2:
                        self._ephemData[key]=subframe[key]
                    self._IODE2 = subframe['IODE2']
                    self._SF2=True
                elif subframe['ID'] == 3 and not self._SF3:
                    for key in ephemSF3:
                        self._ephemData[key]=subframe[key]
                    self._IODE3 = subframe['IODE3']
                    self._SF3=True           
                self._ephemOk = self._SF1 and self._SF2 and self._SF3
                self.EPHEM_LOADED = False

            # in case of live-measurements and reading an out-dated ephemeris from disk, this can cause
            # an error if subframe 4 or 5 set here a time reference. Neither IODC nor weekNum can
            # be checked to ensure that the ephemeris is valid.
            # EPHEM_LOADED=true involves that ephemOk=True since ephemOk is only set false in a new
            # instance with EPHEM_LOADED=false as default.
            if (self._ephemOk and not self.EPHEM_LOADED) or (self.EPHEM_LOADED and subframe['ID'] < 4):
                self._timeData.append((subframe['tow'],subframe['ST']))
                        
        return self._status
    
    
# ----------- Class SatOrbit() ------------------------------------------------------------
# Input for the function readFrame() are dictionairies each containing a single subframe. Many 
# succeeding subframes are passed to an instance of SatData() to build an ephemeris table 
# and a reference time table. A SatPos() instance is used to evaluate this data and to calculate the
# exact positions of the satellite for corrected GPS times using the time table. 
# Feeding the function evalCodePhase() with a list of code phases for given stream numbers
# outputs a list of tuples with satellite positions (x,y,z) at corrected satellite time (tow,n*32ms) and 
# corrected receiving time ST - tupel is (satNo,tow,x,y,z,ST,weekNum,n,cophStd).


class SatOrbit():
    errlst = ['no error',              
              'not ready',
              'new ephem',
              'flawed',
              'unhealthy']
    noErr = 0
    notReady = 1
    newEphem = 2
    flawedFrame = 3
    healthErr = 4
    
    maxSlope = 6.55E-3                 # in units of sample time, 3.2ns/ms; max change of adjacent code phases 
    
    def __init__(self,num,ngps,ncyc,eph=None):
        self._satNo = num
        self._status = 0
        self.data = SatData(num,ephemeris=eph)
        self.position = SatPos()
        self._datLst = []
        self.cpLst = []
        self.NGPS = ngps
        self.NCYC = ncyc
        self.CODESMP = ngps // ncyc
        self.lastSNO = 0
        self.lastCP = 0
        self.REF_TIME = None
        self.REF_EPHEM = None
        self.PHASE_ERR = []              # list of stream numbers with phase error
        self.SCPLst = []                 # list of slopes of code phase curves from streams of length NCYC 
        self.maxSCPLst = 1024//ncyc      # max number o entries in SCPLst (list of slopes over < ncyc code arrays))
        self.minSCPLst = 4               # min number o entries in SCPLst for use of meanSlope
        
    @property 
    def errmsg(self):                
        return self.errlst[self._status]
    
    @property 
    def status(self):        
        return self._status
   
    @property 
    def datLst(self):        
        return self._datLst
    
    @property 
    def satNo(self):        
        return self._satNo       
    
        
    def readFrame(self,subframe):
        streamNo = subframe['ST'] // self.NGPS
        if len(self.PHASE_ERR) > 0 and streamNo < self.PHASE_ERR[-1]:
            self._status = self.flawedFrame
        else:
            self._status = self.data.readSubframe(subframe,saveSubframe=False)        
            if self._status == self.newEphem:
                if self.data.ephemOk:
                    self._datLst.append(self.data)                   # backup previous data
                self.data = SatData(self._satNo)                     #  use a new SatData instance (empty)                
                self.data.readSubframe(subframe,saveSubframe=False)  # then read the subframe again; result is noErr.
            
        return self._status
    
    
    def getStdDev(self,tcpLst,cophLst):
        if len(cophLst) > 3:
            p = np.polyfit(tcpLst,cophLst,1)           # calculate standard deviation by subtracting 
            fit = np.poly1d(p)                         # a linear fit from the data 
            cophStd = np.std(cophLst-fit(tcpLst))  
            self.SCPLst.append(p[0]/self.NCYC)         # scale slope to code positions in stream            
            if len(self.SCPLst) > self.maxSCPLst:
                del self.SCPLst[0]            
        else:
            cophStd = 0.5
            fit = None
            
        cophStd *= GPS_C/SAMPLE_RATE             # standard deviation in m
        
        meanSlope = np.mean(self.SCPLst) if len(self.SCPLst)>self.minSCPLst else 0   # average slopes over ~1000 codes
        if abs(meanSlope) > self.maxSlope:       # max slope is 6.55E-3
            meanSlope = np.sign(meanSlope)*self.maxSlope
        
        return cophStd,meanSlope
            
    
    def clearCodePhaseRef(self):
        self.lastSNO = 0
        self.cpLst = []
        self.SCPLst = []
        self.REF_TIME = None
        self.REF_EPHEM = None
    

    # Checks regarding phase errors should have been done before calling this function!
    def evalCodePhase(self,cpl,relCorr=True):             # cpl = codePhase list of tuples (ST // NGPS, codePhase)
        minGap = 1000                                     # = 32sec 
        maxGap = 10000                                    # = 320sec; rather arbitrary; might be extended (see below)
        minFitNo = self.NCYC // 2
        maxFitNo = 100
        diffTol = 200                                     # depends on maxGap
        
        result = []                    
        if len(cpl) > 0:
            if cpl[0][1] is None:                         # error in measurement occured (jump in codePhase)
                self.PHASE_ERR.append(cpl[0][0])          # save streamNo of error for incoming subframes
                self.data._timeData = []                  # calibrated time lost - reset (TOW,ST)
                self.clearCodePhaseRef()
                return result
            
            cpl = list(filter(lambda item: item[0] > self.lastSNO,cpl))   # ensure that no doubles are in list

        # replace REF_TIME by a measured one based on new ephemerides
        if self.REF_TIME is not None and self.data.ephemOk and self.data.ephemData['IODC'] != self.REF_EPHEM['IODC']:
            self.clearCodePhaseRef()
            
        # timeData != [] involves that data.ephemOk=true (also if ephemeris was loaded, see comments in readSubframe) and
        # that iodc has been checked for valid ephemeris. 
        if self.REF_TIME is None and len(self.data._timeData) > 0:
            self.REF_TIME = self.data._timeData[-1]       # (TOW,ST)
            self.REF_EPHEM = self.data.ephemData.copy()                 
         
        if len(cpl) == 0 or self.REF_TIME is None:      
            return result
        
        weekNum = self.REF_EPHEM['weekNum']
        TOW,ST = self.REF_TIME                            # last valid frame; ST is time of 2048-block with begin of preamble
        ST_DEL = ST % self.CODESMP                        # integer next to codePhase
        ST = (ST // self.CODESMP) * self.CODESMP          # or ST & ~2048; get rid of delay
        ST_SNO = ST // self.NGPS
        
        # ST contains three pieces of information: stream number sno=ST//NGPS, position within stream
        # p=ST//CODESMP-sno*NCYC, and code phase (as integer) with delay = ST % CODESMP.
        # Currently code phases typically arrive < 6sec before the respective ST, since the validity of
        # a subframe can be checked only 6sec after the preamble arrived. 
        # 
        # Code phases are measured without reference to the sample time ST of a subframe (preamble). If
        # the code phase steps over 2048 and thus jumps to 0 (or reverse from 0 to 2048) WITHIN a stream, 
        # this was corrected already (prepCodePhase). If the reference ST is not changed, the following list 
        # of code phases must be corrected here regarding this overflow. This is done using the lastly gathered code 
        # phase lastCP (related stream number is lastSNO). Code phase errors are not seeked - it is assumed that 
        # differences of neighbored code phases (even with gaps) are below diffTol (currently 200).
        
        if ST_SNO > self.lastSNO:                           # at begin and if ref time has been newly set
            self.lastSNO = ST // self.NGPS                  # ST is integer next to interpolated codePhase
            self.lastCP = ST_DEL                           
        
        cpl_SNO,cpl_CP = zip(*cpl)  
        cpl_CP = np.asarray(cpl_CP)
        
        # calculate overflow by a linear fit to previous cplst in case of a large gap of stream numbers.
        if  cpl_SNO[0]-self.lastSNO > maxGap:               # gap too large for fit; reset ref time; currently 10000*32ms = 320sec
            self.clearCodePhaseRef()
            return result                         
        elif cpl_SNO[0]-self.lastSNO > minGap:              # if below minGap=32sec, max drift = 6.6/sec --> change in code phase is < 200
            if len(self.cpLst) >= minFitNo:                 # use at least 16 points for fit, otherwise reset ref time
                x,y = zip(*self.cpLst[-maxFitNo:])          
                p = np.polyfit(x,y,1)                       # linear fit; max. curvature of ~0.7 Hz/s leads to max. mismatch of
                fit = np.poly1d(p)                          # 320sec*0.7Hz/sec =224; diffTol is currently set to 200 (see below)
                self.lastCP = fit(cpl_SNO[0])
            else:
                self.clearCodePhaseRef()
                return result                  

        lastOfl = self.lastCP // self.CODESMP              
        if lastOfl != 0:
            cpl_CP += lastOfl*self.CODESMP                                                                          
                                                           
        diff = self.lastCP - cpl_CP[0]
        if np.isclose(abs(diff),self.CODESMP,rtol=1E-5,atol=diffTol):  # currently diffTol = 200 (see above)
            cpl_CP += np.sign(diff)*self.CODESMP                        
                        
        cophStd,slopeCP = self.getStdDev(cpl_SNO,cpl_CP)  # cophStd ist used as weight in least-squares fit
                                                          # slopeCP is used to correct CP regarding the position in stream   
        
        cpl = list(zip(cpl_SNO,cpl_CP))
        self.cpLst += cpl                                  
        self.lastSNO,self.lastCP = cpl[-1]
        offms = (TOW % 2**(self.NCYC // 32)) * 16 if self.NCYC > 16 else 0   # start offset for given TOW in ms

        # increase TOW (time) up to first stream given in list
        while (ST + 6*SAMPLE_RATE)//self.NGPS < cpl_SNO[0]:  
            ST += 6*SAMPLE_RATE
            TOW += 1
            offms = (offms + 16) % self.NCYC

        CP = cpl_CP[0]                       
        cycNo = 0
        deltaST = offms*self.CODESMP          # in general: deltaST = (offms + cycNo*NCYC)*CODESMP     
        streamNo = (ST+deltaST)//self.NGPS                
        codeNo = (ST+deltaST) // self.CODESMP - streamNo*self.NCYC   # position of code within stream 0 .. NCYC-1
        idx = 0        
        while idx < len(cpl_SNO): 
            if cpl_SNO[idx] < streamNo:       # should not happen
                idx += 1
            elif cpl_SNO[idx] > streamNo:
                streamNo += 1
                cycNo += 1
                deltaST += self.NGPS     
            else:
                x,y,z,d_st = self.position.ecefCoord(TOW,self.REF_EPHEM,DT=deltaST/SAMPLE_RATE,relCorr=relCorr) 
                CP = cpl_CP[idx]              

                corrCP = (codeNo+CP//self.CODESMP-self.NCYC//2)*slopeCP   # correction due to position shift regarding CP measurement (center) 
                smpTime = (ST+deltaST+CP+corrCP)/SAMPLE_RATE + d_st  # in sec; measured codePhase valid for all 2048-blocks in stream
                result.append((self._satNo,TOW,x,y,z,smpTime,weekNum,cycNo,cophStd))  
                streamNo += 1
                cycNo += 1
                deltaST += self.NGPS     
                idx += 1
    
            if deltaST >= 6*SAMPLE_RATE:
                TOW += 1
                cycNo = 0
                ST += 6*SAMPLE_RATE                
                offms = (offms + 16) % self.NCYC
                deltaST = offms*self.CODESMP                
                # following two lines are not required and can be out-commented for debugging
                if streamNo < cpl_SNO[-1]:                          # otherwise REF_TIME might get > lastNo
                    self.REF_TIME = (TOW,ST + CP % self.CODESMP)    # to avoid initial increment of ST; CP needed after reset
                        
        return result
        

# -------- Class SatStream() -------------

class SatStream():
    
    def __init__(self,satNo,freq,itSweep=10,nCyc=32,corrMin=8,corrAvg=8,codeSamples=2048,sampleRate=SAMPLE_RATE,delay=0):
        self.SAT_NO = satNo
        self.CODE_SAMPLES = codeSamples
        self.N_CYC = nCyc
        self.NGPS = nCyc*codeSamples
        self.SEC_TIME = np.linspace(1,self.NGPS,self.NGPS,endpoint=True,dtype=MY_FLOAT)/sampleRate
        self.EDGES = [0]                              # first entry is sign of start bit, subsequent timestamps are in ms and SMP_TIME (tuples)
        self.PHASE_LOCKED = False
        self.PHASE = 0.0         
        self.FREQ = freq          
        self.GPSBITS = np.array([],dtype=np.int8)     # logical bits; 1 = True, -1 = False
        self.GPSBITS_ST = np.array([],dtype=np.int64) # for initialization;  array of SMP_TIME
        self.PREV_SAMPLES = []                       
        self.MS_TIME = 0
        self.SMP_TIME = 0
        self.FFT_CACODE = fft(GPSCacode(satNo,codeSamples))  # Precomputed FFT for inital correlation
        self.DELAY = delay
        self.LASTSEC = 1024 // nCyc
        self.NO_BEFORE_CALC = self.LASTSEC           # calculate once in a second
        self.CORR_MIN = corrMin                      # currently 8,  threshold for determination of valid correlation maximum 
        self.CORR_AVG = min(corrAvg,nCyc)            # currently 8; number of CODE_SAMPLES to average in FFT (<= N_CYC)
        self.SWEEP_CORR_AVG = 4                      # the same for sweep, here high precision not needed
        self.MIN_FREQ = -5000.0
        self.MAX_FREQ = +5000.0
        self.STEP_FREQ = 200                         # for sweepFreq
        self.CACODE_REP = GPSCacodeRep(satNo,codeSamples,nCyc,0)
        self.STD_DEV = 0.005                        # overwritten with first stream
        self.AMPLITUDE = 0.0
        self.MAX_CORR = 0.0
        self.SWEEP = False
        self.IT_SWEEP = itSweep
        self.PREV_STREAM_NO = 0
        self.PREV_SIGNAL = 0                        # for determining edges in decodeData
        self.DF_GAIN1 = 10                          # before phaseLock
        self.DF_GAIN2 = 1                           # after phaseLock
        self.DF_NO = self.LASTSEC                   
        self.DF = [0]
        self.PH_0 = True
        self.CORR_Q = 0                             # -1..+1; mean of corr results of last 60 sec (CORRLST_NO)
        self.CORR_L = 0                          # similar to CORR_Q but only last second
        self.CORRLST_NO = 60*self.LASTSEC             
        self.CORRLST = [0]                          # saves codePhases to calculate quality (CORR_Q)
        self.MIN_CORR_Q = -0.9
        self.REP_SWEEP = False
        self.CALC_PLOT = False                     # set True if BITDATA and GPSDATA shall be calculated
        self.GPSDATA = []                          # for external access if data shall be plotted
        self.BITDATA = []                          # (dito)
    
    # ---------------
    
    def erasePrevData(self):
        self.EDGES = [0]             
        self.GPSBITS = np.array([],dtype=np.int8)     
        self.GPSBITS_ST = np.array([],dtype=np.int64) 
        self.PREV_SAMPLES = []                                   
        
    
    def setPhaseUnlocked(self):
        self.PHASE_LOCKED = False
        self.CORRLST = [0]
        self.MS_TIME = 0
        self.PHASE = 0.0         
        self.erasePrevData()
        
        
    def initSweep(self):
        self.setPhaseUnlocked()            
        self.FREQ_SAVE = self.FREQ      # backup if sweep fails
        self.DF_SAVE = self.DF.copy()
        self.FREQ = self.MIN_FREQ
        self.DF = [0]
        self.SWEEP = True
        
    def restoreFreq(self):
        self.FREQ = self.FREQ_SAVE
        self.DF = self.DF_SAVE.copy()        
    
    # ---------------
    
    def reportValues(self,frameLst):                      # called in process()
        for dct in frameLst:
            dct['SAT'] = self.SAT_NO        
            dct['AMP'] = self.AMPLITUDE
            dct['CRM'] = self.MAX_CORR
            dct['FRQ'] = self.FREQ
            dct['SWP'] = self.REP_SWEEP     
            
        self.REP_SWEEP = False            

            
    def checkCorrQuality(self):                        # called in process()
        if len(self.CORRLST) == self.CORRLST_NO:       # min of 60 sec between sweeps
            sweep = (self.CORR_Q < self.MIN_CORR_Q)
        else:
            sweep = False
        return sweep            

    
    def process(self,data,smpTime,sweep=False):  
        self.SMP_TIME = smpTime
        streamNo = smpTime // self.NGPS
        if streamNo -1 != self.PREV_STREAM_NO:
            self.erasePrevData()
        self.PREV_STREAM_NO = streamNo            
        sweep = sweep and not self.SWEEP                   # ignore sweep trigger if SWEEP is already running  
        
        if sweep:                    
            self.initSweep()                               # sets SWEEP = True and FREQ = MIN_FREQ
        
        if not self.SWEEP:       
            data,self.PHASE = self.demodDoppler(data,self.FREQ,self.PHASE,self.NGPS)    
            _,delay,codePhase,normMaxCorr = self.cacodeCorr(data,self.CORR_AVG)
            self.CORR_Q,self.CORR_L = self.corrQuality(codePhase)
            
            if delay >= 0:
                self.DELAY = delay                          # DELAY is updated if correlation is above threshold
            gpsData = self.decodeData(data,self.DELAY)      # self.EDGES saves events of signal if PHASE_LOCKED 
                                                            # gpsData has N_CYC complex data points (1 per ms)
            self.STD_DEV = np.std(np.abs(gpsData))
            self.AMPLITUDE = np.mean(np.abs(gpsData))/self.STD_DEV
            self.MAX_CORR = normMaxCorr                       

            frameLst = []
            if streamNo % self.NO_BEFORE_CALC == 0:         # process send data once in a second                       
                if self.PHASE_LOCKED:
                    frameLst = self.evalEdges()             # list of subframe data                     
                if len(frameLst) == 0:
                    frameLst = [{}]                                            
                self.reportValues(frameLst)
                sweep = self.checkCorrQuality()                    
                    
            if sweep:                    
                self.initSweep()
            else:
                dfreq,phaseshift,self.PHASE_LOCKED,phase = self.phaseLockedLoop(gpsData)  
                self.PHASE += phaseshift
                self.FREQ  = self.confineRange(self.FREQ + dfreq)
                
            return self.SWEEP,frameLst,codePhase,(self.CORR_Q,self.CORR_L)   # SWEEP: return what will be done next
        
        else:   # self.SWEEP = True
            
            self.REP_SWEEP = self.SWEEP                # for report in dct
            
            if self.SWEEP:
                self.SWEEP,self.FREQ,self.MAX_CORR,delay,codePhase = self.sweepFrequency(data,self.FREQ)
                if not self.SWEEP and delay < 0:    # sweep failed
                    self.restoreFreq()              # restore backup frequency (set in initSweep)
            if delay > -1:
                self.DELAY = delay                    
            self.CORR_Q,self.CORR_L = self.corrQuality(codePhase)    
            
            frameLst = []
            if streamNo % self.NO_BEFORE_CALC == 0:        # process send data once in a second                       
                frameLst = [{}]            
                self.reportValues(frameLst)                    
                self.REP_SWEEP = self.SWEEP

            return self.SWEEP,frameLst,codePhase,(self.CORR_Q,self.CORR_L)     # SWEEP: return what will be done next
                
    # ---------------
    
    def phaseLockedLoop(self,gpsData):
        avg = 4       
        minDiff = 2.0                               # instead of a jump of 2, a jump of pi-2 = 1.4 is more probable
        lockedThres = 0.1      
        minAmp = 3
        maxDF = 20/(1024//self.N_CYC)               # max 20 Hz/sec; largest change measured for good signal was 2.64                                                   
                                                    # 10 was good, but to slow at beginning
                                                    # prevents fast frequency change for bad signals            
        phaseIsLocked = self.PHASE_LOCKED            # to avoid changing global variable in this function
        n = len(gpsData)                             # currently n = NCyl = 64; 1 point/ms
        phase = np.arctan(gpsData.imag/gpsData.real) # removes modulation, to determine phase (-pi/2 to pi/2)
        
        dp = 0
        realPhase = np.copy(phase)
        for i in range(1,n):                         # try to rebuild real phase change
            delta = phase[i]-phase[i-1]
            dp -= np.sign(delta) if abs(delta) > minDiff else 0            
            realPhase[i] += dp*np.pi

        phOff = np.mean(realPhase[-avg:])                       
        phaseDev = np.mean(realPhase)          

        # feedback control to keep phase close to 0 (imag -> 0) by finding correction df to doppler frequency 
        # phaseDev averages over signal; important to suppress imaginary part
        # ph_0=True resets phase to zero and thereby selects real part
        # "memory" DF allows temporal switching-off fast feedback - good signal/noise, stable locking.
        # Low signal causes large phase changes and thus unreliable feedback signal, therefore
        # max freq. change maxDF has been introduced        
        if phaseIsLocked:                            # currently gain2 = 1, gain1 = 10
            meanDF = np.mean(self.DF)
            fbDF = self.DF_GAIN2*phaseDev            # if self.AMPLITUDE > minAmp else 0
            df = fbDF + meanDF                       # weight can be added, e.g. (meanDF*DF_NO + fbDF)/(DF_NO+1)
            if abs(df) > maxDF:
                df = np.sign(df)*maxDF
            if len(self.DF) >= self.DF_NO:
                del self.DF[0]
            self.DF.append(df)
        else:
            df = self.DF_GAIN1*phaseDev               
            self.DF = [df]

        if abs(phaseDev) < lockedThres:
            phaseIsLocked = True

        phaseOffset = phOff if self.PH_0 else 0

        return df, phaseOffset, phaseIsLocked, realPhase

        
    # ---------------

    
    def fitCodePhase(self,gpsCorr,mx):
        lgc = len(gpsCorr)
        ma = mx - 1 if mx > 0 else lgc-1
        mb = mx + 1 if mx < lgc-1 else 0

        # Fit to triangle 
        if gpsCorr[ma] > gpsCorr[mb]:
            tmx = 0.5*(gpsCorr[mb]-gpsCorr[ma])/(gpsCorr[mx]-gpsCorr[mb])
        else:
            tmx = 0.5*(gpsCorr[mb]-gpsCorr[ma])/(gpsCorr[mx]-gpsCorr[ma])
        
        # Fit to parabola, x-value for minimum    
        pmx = 0.5*(gpsCorr[mb]-gpsCorr[ma])/(2*gpsCorr[mx]-gpsCorr[mb]-gpsCorr[ma])   # fmx from [-0.5,+0.5]
        
        fitMax = mx + 0.5*(tmx+pmx)                # mean of both fits gives best value (from tests); ranges from -0.5 to 2047.5
            
        return fitMax


    def findCodePhase(self,gpsCorr):
        mean = np.mean(gpsCorr)
        std = np.std(gpsCorr)
        delay = -1
        codePhase = -1.0
        mx = np.argmax(gpsCorr)
        normMaxCorr = (gpsCorr[mx]-mean)/std
        if normMaxCorr > self.CORR_MIN:   
            delay = mx
            codePhase = self.fitCodePhase(gpsCorr,mx)   
            
        return delay,codePhase,normMaxCorr

    # ---------------
        
    # The position of the correlaton peak determined here depends on which of the FFT-transformed arrays
    # (fftData or FFT_CACODE) is conjugated for the back transformation. Here, if the position of the first bit of 
    # the cacode sequence in the data is shifted to the right by DS, the correlation peak is at DELAY = DS, 
    # which is the returned value in correlation. Perfect alignment is achieved if the cacode is rolled to the 
    # right by DELAY=DS, np.roll(cacode,+DELAY). This is done in decodeData for decoding. 
    def cacodeCorr(self,data,corrAvg): 
        df = 0
        ncyc = len(data)//self.CODE_SAMPLES
        p = (ncyc-corrAvg) // 2                           # correlation is done in center of data
        for i in range(p,p+corrAvg):
            dfm = fft(data[i*self.CODE_SAMPLES:(i+1)*self.CODE_SAMPLES])
            df += dfm
        fftData = df/corrAvg
        fftCorr = fftData*np.conjugate(self.FFT_CACODE)   # FFT_CACODE computed once in init
        corr = np.abs(ifft(fftCorr))
        delay,codePhase,normMaxCorr = self.findCodePhase(corr)
        return corr,delay,codePhase,normMaxCorr

    # ---------------

    def corrQuality(self,codePhase):
        cpq = -1 if codePhase < 0 else 1 
        self.CORRLST.append(cpq)
        if len(self.CORRLST) > self.CORRLST_NO:           # currently 60*(1024//N_CYC) 
            del self.CORRLST[0]
        corrQ = np.mean(self.CORRLST)
        corrLast = np.mean(self.CORRLST[-self.LASTSEC:]) 

        return corrQ,corrLast 
                
    # --------------- 
        
    def demodDoppler(self,data,freq,phase,N):                          
        factor = np.exp(-1.j*(phase+2*np.pi*freq*self.SEC_TIME[:N]))
        phase += 2*np.pi*freq*self.SEC_TIME[N-1]
        return factor*data[:N], np.remainder(phase,2*np.pi)

    #------------------

    def getCorrMax(self,data,corrAvg,freq):
        phase = 0
        N = corrAvg*self.CODE_SAMPLES
        new1,_ = self.demodDoppler(data,freq,phase,N)
        corr1,delay,coPh,normMaxCorr = self.cacodeCorr(new1,corrAvg)   # codePhase ignored for sweep
        if delay > -1:
            max1 = corr1[delay]
        else:
            max1 = 0
        return delay,max1,coPh,normMaxCorr


    def sweepFrequency(self,data,freq): 
        sweepFreq = True
        j = 0
        delay = -1    
        coPh = -1
        while delay < 0 and j < self.IT_SWEEP:   
            delay,maxCorr,coPh,normMaxCorr = self.getCorrMax(data,self.SWEEP_CORR_AVG,freq) 
            if delay < 0: 
                freq = freq+self.STEP_FREQ 
            j += 1            
            
        if delay >= 0:
            sweepFreq = False                        
        elif freq > self.MAX_FREQ:
            freq = self.MIN_FREQ
            sweepFreq = False
        
        return sweepFreq,freq,normMaxCorr,delay,coPh  # previously maxCorr


    # ---------------
        
    # In the correlation a delay is determined ranging from 0 to 2047. The cacode is shifted (rolled) right 
    # to decode the data through multiplication. The start bit of the cacode in the data array is at 
    # SMP_TIME + delay. In prevSample, it is at position SMP_TIME+delay-2048. 
    def decodeData(self,data,delay): 
        prevSign = (2*(len(self.EDGES) % 2) - 1) * self.EDGES[0]     # recalculation of last sign in EDGES

        cacode = np.roll(self.CACODE_REP,delay) # not optimal; obs time and sat time can change up to 0.2 samptime units 
                                                # in 32ms (3 cacode chips in a second) and codePhase is no integer
        y = cacode*data                                     

        NPS = len(self.PREV_SAMPLES)            # if delay has not changed compared to previous stream, NPS + delay = 2048
        if NPS > 0:
            y = np.append(self.PREV_SAMPLES,y)
        NS = self.NGPS + NPS                    # new length of data array                              

        n0 = 0               
        n1 = NPS + delay                        # for first avg sometimes n1-n0 != 2048; n=NPS is at data[0]  
        if n1 == 0:
            n1 = self.CODE_SAMPLES
            ST = self.SMP_TIME
        else:        
            ST = self.SMP_TIME + delay - self.CODE_SAMPLES  # used for saving edge events in local time (SDR clock)
        
        gpsData = []
        while n1 <= NS:
            m = np.mean(y[n0:n1])
            gpsData.append(m)
            if self.PHASE_LOCKED: 
                mSign = np.sign(m.real)
                if self.EDGES[0] == 0:                   # no entries
                    self.EDGES[0] = mSign                # first entry is sign of first signal
                    prevSign = mSign             
                else:
                    if mSign != prevSign \
                    and prevSign*self.PREV_SIGNAL > 0 \
                    and abs(m.real-self.PREV_SIGNAL) > 3*self.STD_DEV:  
                        self.EDGES.append((self.MS_TIME,ST+n0))   # timestamps of sign change in ms and sample time
                        prevSign = mSign
                self.PREV_SIGNAL = m.real
                self.MS_TIME += 1                       # satellite time, increased at cacode begin    
            n0 = n1
            n1 += self.CODE_SAMPLES
        gpsData = np.asarray(gpsData,dtype=MY_COMPLEX) 
        self.PREV_SAMPLES = y[n0:NS]

        if self.CALC_PLOT:
            self.GPSDATA = gpsData
            self.BITDATA = self.bitPlotData(len(gpsData))    # uses self.EDGES

        return gpsData                                       

    # ---------------
    
        
    def evalEdges(self):                  # processed once in a second (see process() )   
        frameData = []
        
        if len(self.EDGES) > 2:            
            bits,bitsSmpTime = self.logicalBits()       
            self.GPSBITS = np.append(self.GPSBITS,bits)
            self.GPSBITS_ST = np.append(self.GPSBITS_ST,bitsSmpTime)   
            frameData,self.GPSBITS,self.GPSBITS_ST = self.evalGpsBits(self.GPSBITS,self.GPSBITS_ST) 

        return frameData        
    
    # ---------------
            
    def logicalBits(self):             # input is stream of EDGES, tuple of MS_TIME and SMP_TIME
        bits = []
        bitsSmpTime = []
        lastSign = self.EDGES[0]
        n = len(self.EDGES)

        if n > 2:
            t1,st1 = self.EDGES[1]
            for i in range(2,n):
                t2,st2 = self.EDGES[i]
                m,r = np.divmod(t2-t1,20)
                if r > 17:
                    m += 1
                if m > 0:                
                    bits += [lastSign]*m
                    bitsSmpTime += [st1]    # only save sample time at edges
                    bitsSmpTime += [0]*(m-1) # add 0 so that bitsSmpTime have same index as bits
                t1 = t2
                st1 = st2
                lastSign = -lastSign        
            self.EDGES = [lastSign,self.EDGES[-1]]  # previous data of EDGES no longer needed        

        bits = np.asarray(bits,dtype=np.int8)     # caution: logical False is here -1 
        bitsSmpTime = np.asarray(bitsSmpTime,dtype=np.int64)

        return bits, bitsSmpTime

    
    #------------- Function evalGpsBits (uses Subframe) -----------------------------------    
    #
    # Reads a stream of bits (gpsBits with +1 and-1), finds the locations of the preamble, 
    # and extracts the data using an instance of Subframe(). The data of input arrays not 
    # decoded in this process are given back as result together with a dictionairy containing 
    # the subframe parameter. The data ST from gpsBitsSmpTime for the start of the preamble 
    # is put into the subframe data. 

    def evalGpsBits(self,gpsBits,gpsBitsSmpTime):    
        Result = []
        if len(gpsBits) < 300:
            return Result, gpsBits, gpsBitsSmpTime

        gb = np.copy(gpsBits)
        preamble = np.array([1,-1,-1,-1,1,-1,1,1],dtype=np.int8)
        bitsCorr = np.correlate(gb,preamble,mode='same')
        locPreamble =[]
        for i in range(len(bitsCorr)):
            if abs(bitsCorr[i])==8:
                locPreamble.append(i-4)     # begin of preamble 4 bits before maximum in corr

        start = 0
        if len(locPreamble) > 0: 
            gb[gb==-1] = 0                  # convert to logical numbers (-1 to 0)
            lpIndex = 0
            start = locPreamble[lpIndex]
            ok = True
            while ok and start+300 < len(gb):  # ok is True if start is changed below
                sf = Subframe()  
                if sf.Extract(gb[start:start+300]) == 0:
                    ST = gpsBitsSmpTime[start]
                    if sf.ID == 1:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'weekNum': sf.weekNum,
                               'satAcc': sf.satAcc,
                               'satHealth': sf.satHealth,
                               'Tgd': sf.Tgd,
                               'IODC': sf.IODC,
                               'Toc': sf.Toc,
                               'af2': sf.af2,
                               'af1': sf.af1,
                               'af0': sf.af0,
                               'ST': ST}
                    elif sf.ID == 2:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'Crs': sf.Crs,
                               'deltaN': sf.deltaN,
                               'M0': sf.M0,
                               'Cuc': sf.Cuc, 
                               'IODE2': sf.IODE2,
                               'e': sf.e,
                               'Cus': sf.Cus,
                               'sqrtA': sf.sqrtA,
                               'Toe': sf.Toe,
                               'ST': ST}
                    elif sf.ID == 3:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'Cic': sf.Cic,
                               'omegaBig': sf.omegaBig,
                               'Cis': sf.Cis,
                               'i0': sf.i0, 
                               'IODE3': sf.IODE3,
                               'Crc': sf.Crc,
                               'omegaSmall': sf.omegaSmall,
                               'omegaDot': sf.omegaDot,
                               'IDOT': sf.IDOT,
                               'ST': ST}
                    elif sf.ID == 4 or sf.ID == 5:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'ST': ST}
                    Result.append(res)
                    start += 300  
                else:
                    ok = False
                    while not ok and lpIndex<len(locPreamble)-1:
                        lpIndex += 1
                        s = locPreamble[lpIndex]
                        ok = s > start                    
                    if ok:
                        start = s                

        return Result, gpsBits[start:], gpsBitsSmpTime[start:]

    
    # -----------------

    def bitPlotData(self,n):              # n is requested length of data 
        bd = np.zeros(n,dtype=np.int8)    # bit Data for plot with n points
        t1 = self.MS_TIME                 # last time of array
        t0 = t1 - n + 1                   # start time, total time period is 32 ms (N_CYC)
        firstSign = self.EDGES[0]
        
        if len(self.EDGES) == 1:
            bd[:] = firstSign
        elif len(self.EDGES) == 2:            
            t,st = self.EDGES[1]
            if t >= t0:
                bd[:t-t0] = firstSign
                bd[t-t0:n] = -firstSign
            else:
                bd[:] = -firstSign
        else:
            dt = self.EDGES[-1][0] - self.EDGES[1][0] 
            lastSign = (2*(len(self.EDGES) % 2) - 1) * self.EDGES[0] 
            bsc = [tms for tms,st in self.EDGES[1:]]  # array of only ms times
            k1 = len(bsc)
            k0 = k1-1
            while k0>0 and bsc[k0]>t0:
                k0 -= 1
            ts = 0
            for t in reversed(bsc[k0:k1]):  
                te = min(ts+t1-t, n)
                bd[ts:te] = lastSign
                ts = te
                t1 = t
                lastSign = -lastSign
            if te < n:
                bd[te:n] = lastSign
                
        bd = np.flip(bd)

        return bd

    # ---------------
            
    def confineRange(self,freq):
        if freq > self.MAX_FREQ:
            freq = self.MAX_FREQ
        elif freq < self.MIN_FREQ:
            freq = self.MIN_FREQ
        return freq        
    
    
# -------- end class SatStream() ----------    
    

# ----------- Positioning by using the Newton-Gauss algorithm -----------------------------
    

# for 4 or more satellites
def JacobianCalc(pos,satPos,rangeEst):
    jacob = np.zeros((np.size(satPos,1),4))
    jacob[:,0] = -np.ones(np.size(satPos,1))         # partial derivative regarding c*t0
    jacob[:,1] = (pos[1] - satPos[0,:])/rangeEst   # partial derivatives regarding pos(x,y,z)
    jacob[:,2] = (pos[2] - satPos[1,:])/rangeEst
    jacob[:,3] = (pos[3] - satPos[2,:])/rangeEst
    
    return jacob

# for 3 or more satellites; height h is fixed by x^2+y^2+z^2 = (R+h)^2 with R given by earth radius
def JacobianCalc3(pos,satPos,rangeEst):
    a = 6378137                                     # m; WGS84 major axis
    f = 1/298.257223563                             # = 1/f; WGS84 inverse flattening: a/b = 1/(1-f) = if/(if-1)
    ab2 = 1/(1-f)**2                                # = (a/b)^2
    absPos = np.sqrt(pos[1]**2+pos[2]**2+ab2*pos[3]**2)
    jacob = np.zeros((np.size(satPos,1)+1,4))
    jacob[:-1,0] = -np.ones(np.size(satPos,1))      # partial derivative regarding t0
    jacob[:-1,1] = (pos[1] - satPos[0,:])/rangeEst  # partial derivatives regarding pos(x,y,z)
    jacob[:-1,2] = (pos[2] - satPos[1,:])/rangeEst
    jacob[:-1,3] = (pos[3] - satPos[2,:])/rangeEst
    jacob[-1,0] = 0
    jacob[-1,1] = pos[1]/absPos
    jacob[-1,2] = pos[2]/absPos
    jacob[-1,3] = ab2*pos[3]/absPos

    return jacob


def rotEarth(recPos,rangeEst):
    dt = rangeEst/GPS_C
    v = [-recPos[2]*OMEGA_EARTH,recPos[1]*OMEGA_EARTH,0]
    dp = np.tensordot(v,dt,0)

    return dp

# see https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
# and https://en.wikipedia.org/wiki/Weighted_least_squares
#
# position of receiver and propagation time t0 of Satellite 0 is calculated
# from known satellite positions and measured time delays between subframes
# number of satellites, i.e. dimension of satPos, must be n >= 4
#
# satPos is np.array([x,y,z],:noSats) in ECEF coordinates
# recPos is receiver position [t0,x,y,z] to be determined

def leastSquaresPos(minSat,satPos,timeDelay,**kwargs):  
    if minSat == 4:
        recPos,residLst,rangeEst,measDelay = leastSquaresPos4(satPos,timeDelay,**kwargs)
    else:
        recPos,residLst,rangeEst,measDelay = leastSquaresPos3(satPos,timeDelay,**kwargs)    
        
    return recPos,residLst,rangeEst,measDelay

      
def leastSquaresPos4(satPos,timeDelay,recPos=[0,0,0,0],maxResidual=1.0E-8,maxIt=10,\
                       t0Guess=0.07,height=150,hDev=1,stdDev=None):     
    
    dt = timeDelay - timeDelay[0]                   # difference times regarding t0 (measured values)    
    cdt = GPS_C*dt
    recPos[0] = GPS_C*t0Guess                       # 4th free variable 
    residLst = []
    dp = np.zeros((3,len(dt)))
    
    if stdDev is None:
        W = np.eye(len(dt))
    else:
        W = np.linalg.inv(np.diag(stdDev)**2)      # weight matrix: 1/sigma^2 in diagonal; sigma in m
    
    it = 0
    residual = 1
    while it < maxIt and residual > maxResidual:    
        rangeEst = np.sqrt((satPos[0,:]-recPos[1]-dp[0,:])**2 + (satPos[1,:]-recPos[2]-dp[1,:])**2 + (satPos[2,:]-recPos[3]-dp[2,:])**2) 

        # displacement of recorder position due to rotation of earth
        dp = rotEarth(recPos,rangeEst)
        
        # function to minimize using Gauss-Newton method; free variables are in recPos
        fgn = rangeEst - recPos[0] - cdt
        
        Jac = JacobianCalc(recPos,satPos,rangeEst)
        JacT = Jac.T    
    
        deltaPos = -np.linalg.pinv(JacT.dot(W).dot(Jac)).dot(JacT).dot(W) @ fgn   #  pinv is 2x slower than inv
        recPos += deltaPos                                                        #  (~ 0.4ms instead of 0.2ms) 
                
        residual = np.linalg.norm(deltaPos)
        residLst.append(residual)
        it += 1
        
    measDelay = cdt + recPos[0]                     # propagation times of satellites (in m) 
            
    return recPos,residLst,rangeEst,measDelay


# for 3 or more Satellites; z-position pos[3] is fixed by x^2+y^2+z^2 = R^2 with R given by earth radius 
def leastSquaresPos3(satPos,timeDelay,recPos=[0,3687000,3687000,0],maxResidual=1.0E-8,maxIt=10,\
                       t0Guess=0.07,height=150,hDev=1,stdDev=None):     
    a = 6378137                                     # m; WGS84 major axis
    f = 1/298.257223563                             # = 1/f; WGS84 inverse flattening: a/b = 1/(1-f) = if/(if-1)
    ab2 = 1/(1-f)**2                                # = (a/b)^2    

    dt = timeDelay - timeDelay[0]                   # difference times regarding t0 (measured values)    
    cdt = GPS_C*dt
    recPos[0] = GPS_C*t0Guess                          
    recPos[3] = (1-f)*np.sqrt((a+height)**2-recPos[1]**2-recPos[2]**2)
    recPos = np.asarray(recPos)
    fgn = np.zeros(len(dt)+1)
    residLst = []
    dp = np.zeros((3,len(dt)))    

    if stdDev is None:
        W = np.eye(len(dt)+1)
    else:
        stdDev = np.append(stdDev,[hDev])
        W = np.linalg.inv(np.diag(stdDev)**2)      # weight matrix: 1/sigma^2 in diagonal; sigma in m

    it = 0
    residual = 1
    while it < maxIt and residual > maxResidual:    
        rangeEst = np.sqrt((satPos[0,:]-recPos[1]-dp[0,:])**2 + (satPos[1,:]-recPos[2]-dp[1,:])**2 + (satPos[2,:]-recPos[3]-dp[2,:])**2) 

        # displacement of recorder position due to rotation of earth
        dp = rotEarth(recPos,rangeEst)
        
        # functions to minimize using Gauss-Newton method; free variables are in pos
        fgn[:-1] = rangeEst - recPos[0] - cdt
        fgn[-1] = np.sqrt(recPos[1]**2+recPos[2]**2+ab2*recPos[3]**2)-(a+height)    # earth ellipsoid (x^2+y^2)/a^2 + z^2/b^2 = 1

        Jac = JacobianCalc3(recPos,satPos,rangeEst)
        JacT = Jac.T    

        deltaPos = -np.linalg.pinv(JacT.dot(W).dot(Jac)).dot(JacT).dot(W) @ fgn 
        recPos += deltaPos        

        residual = np.linalg.norm(deltaPos)
        residLst.append(residual)
        it += 1

    measDelay = cdt + recPos[0]                       # measured propagation times of satellites (in m)
            
    return recPos,residLst,rangeEst,measDelay
                        


# ------ Transformation ECEF coordinates to geodetic coordinates (latitude,longitude, altitude) ----
# Karl Osen. Accurate Conversion of Earth-Fixed Earth-Centered Coordinates to Geodetic Coordinates.
# [Research Report] Norwegian University of Science and Technology. 2017. hal-01704943v2
# (half as fast as pyproj.transformer.transform(), but more accurate)
# short names
invaa    = +2.45817225764733181057e-0014   # 1/(a^2) 
aadc     = +7.79540464078689228919e+0007    # (a^2)/c 
bbdcc    = +1.48379031586596594555e+0002   # (b^2)/(c^2) 
l        = +3.34718999507065852867e-0003    # (e^2)/2 
p1mee    = +9.93305620009858682943e-0001   # 1-(e^2) 
p1meedaa = +2.44171631847341700642e-0014 # (1-(e^2))/(a^2) 
Hmin     = +2.25010182030430273673e-0014    # (e^12)/4 
ll4      = +4.48147234524044602618e-0005    # e^4 
ll       = +1.12036808631011150655e-0005  # (e^4)/4 
invcbrt2 = +7.93700525984099737380e-0001 # 1/(2^(1/3)) 
inv3     = +3.33333333333333333333e-0001    # 1/3 
inv6     = +1.66666666666666666667e-0001    # 1/6 
d2r      = +1.74532925199432957691e-0002     # pi/180 
r2d      = +5.72957795130823208766e+0001     # 180/pi 

def geoToEcef(lat,lon,alt):
    lat = d2r * lat
    lon = d2r * lon
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    N = aadc/np.sqrt(coslat * coslat + bbdcc)
    d = (N + alt) * coslat
    x = d * coslon
    y = d * sinlon
    z = (p1mee * N + alt) * sinlat
    return x,y,z


def ecefToGeo(coordinates):
    x,y,z = coordinates
    ww = x * x + y * y
    m = ww * invaa
    n = z * z * p1meedaa
    mpn = m + n
    p = inv6 * (mpn - ll4)
    G = m * n * ll
    H = 2 * p * p * p + G
    if (H < Hmin):
        return None
    C = math.pow(H + G + 2 * np.sqrt(H * G), inv3) * invcbrt2
    i = -ll - 0.5 * mpn
    P = p * p
    beta = inv3 * i - C - P / C
    k = ll * (ll - mpn)
    
    t1 = beta * beta - k
    t2 = np.sqrt(t1)
    t3 = t2 - 0.5 * (beta + i)
    t4 = np.sqrt(t3)
    
    t5 = np.abs(0.5 * (beta - i))
    # t5 may accidentally drop just below zero due to numeric turbulence
    # This only occurs at latitudes close to +- 45.3 degrees
    t6 = np.sqrt(t5)
    t7 = t6 if (m < n) else -t6
    
    t = t4 + t7
    # Use Newton-Raphson's method to compute t correction
    j = l * (m - n)
    g = 2 * j
    tt = t * t
    ttt = tt * t
    tttt = tt * tt
    F = tttt + 2 * i * tt + g * t + k
    dFdt = 4 * ttt + 4 * i * t + g
    dt = -F / dFdt
    # compute latitude (range -PI/2..PI/2)
    u = t + dt + l
    v = t + dt - l
    w = np.sqrt(ww)
    zu = z * u
    wv = w * v
    lat = np.arctan2(zu, wv)
    # compute altitude
    invuv = 1 / (u * v)
    dw = w - wv * invuv
    dz = z - zu * p1mee * invuv
    da = np.sqrt(dw * dw + dz * dz)
    alt = -da if (u < 1) else da
    # compute longitude (range -PI..PI)
    lon = np.arctan2(y, x)
    
    lat = r2d * lat
    lon = r2d * lon
    
    return lat,lon,alt


# ----- Transformation from ECEF coordinates to local azimuth & elevation of satellites ---
#
# input: observer position [x,y,z] (ECEF), satellite position [x,y,z]
# output : elevation & azimuth (theta,phi) in degree (north = 0, east = 90° etc)

def ecefToAzimElev(obsPos,satPos):
    r1    = np.asarray(obsPos)           # vector to observer
    mr1   = np.sqrt(np.dot(r1,r1))       # magnitude of r1; faster than np.linalg.norm()
    r2    = np.asarray(satPos)           # vector to satellite 
    r21   = r2 - r1                      # difference vector
    mr21  = np.sqrt(np.dot(r2-r1,r2-r1)) # magnitude of r2-r1
    n1    = r1 / mr1                     # normal vector of local area at obsPos
    r21p  = np.dot(n1,r2-r1)*n1          # projection of r2-r1 to n1
    r21e  = r2 - r1 - r21p               # projection of r2-r1 to area
    mr21e = np.sqrt(np.dot(r21e,r21e))   # magnitude of projection of r2-r1 to area
    z1    = np.asarray([0,0,1])          # direction 'north' 
    z1p   = np.dot(z1,n1)*n1             # projection of z1 to n1
    z1e   = z1 - z1p                     # projection of z1 to area
    mz1e  = np.sqrt(np.dot(z1e,z1e))     # magnitude of projection of z1 to area

    # scalar product of n1 and r21/mr21 is sine of elevation angle theta, thus
    theta = np.arcsin(np.dot(n1,r21)/mr21)/np.pi*180

    # scalar product of z1e and r21e gives azimuth angle phi, thus
    phi = np.arccos(np.dot(z1e,r21e)/(mz1e*mr21e))/np.pi*180
    
    # triple product (Spatprodukt) determines sign of angle
    if np.dot(n1,np.cross(r21e,z1e)) < 0:
        phi = -phi    

    return theta,phi


# ---- calc distances in m from local position in geodetic coordinates (latitude & longitude)
# based on ellipsoidal model of earth (WGS84)
# input: lonHome, latHome is local position (in degree)
#        lon,lat is in a (small) distance to "Home" (in degree)
# output: delta x in m (along latitude to east), delty y in m (along longitude to north)

def locDistFromLatLon(geoHome,geoPos):  # geoCoord = (lat,lon,height)
    eqAxis =  6378137.0
    flat =  0.003352810

    latHome,lonHome,_ = geoHome
    lat,lon,_ = geoPos
    lonDistPerDeg = eqAxis*(np.sin(latHome/180*np.pi)**2+((1-flat)*np.cos(latHome/180*np.pi))**2)**(3/2)/(1-flat)*np.pi/180
    latDistPerDeg = eqAxis*np.cos(latHome/180*np.pi)*np.pi/180

    return (lon-lonHome)*latDistPerDeg, (lat-latHome)*lonDistPerDeg

# ------ time string from GPS week number and time-of-week --------------------------------------

def gpsTime(tow,weekNum):
    leaptime = 18          # current difference of GPS and UTC time
    rollover = 2
    tow = getattr(tow, "tolist", lambda: tow)()          # if numpy types convert it to native python 
    weekNum = getattr(weekNum, "tolist", lambda: weekNum)()
    # tow is valid for start of next subframe; tow-1 for current subframe
    # tow is np.int32; converted to int of python with tow.item()
    d = datetime.datetime(1980,1,6) + datetime.timedelta(days=(weekNum+rollover*1024)*7) \
        +datetime.timedelta(seconds=(tow-1)*6-leaptime)
    
    return d
    
    
def gpsTimeStr(tow,weekNum,timeOnly = False):    
    d = gpsTime(tow,weekNum)    
    if timeOnly:
        return d.strftime('%H:%M:%S UTC') 
    else:
        return d.strftime('%a, %d.%m.%Y %H:%M:%S UTC')  
    

    
