# Horton's method forward

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.header("Infiltration and rainfall excess calculation")
st.latex(r'''  f = f_c + (f_o-f_c)\exp(-kt) ''')
st.latex(r'''  F = f_c t + (f_o-f_c)\left[\frac{1-\exp(-kt)}{k}\right] ''')
st.markdown("Where,")
st.markdown("$f$ = Actual infiltratoin rate &emsp; [mm min$^{-1}$]")
st.markdown("$f_c$ = Steady sate infiltration rate &emsp; [mm min$^{-1}$]")
st.markdown("$f_o$ = Infiltration rate just gegining of rainfall &emsp; [mm min$^{-1}$]")
st.markdown("$k$ = Horton's decay consant &emsp; [min$^{-1}$]")
st.markdown("$F$ = Potential cumulative Infiltration &emsp; [mm]")

# If we can make the plots with plotly it will be easy to get the values

with st.sidebar:
    
    st.header("Precipitation data")
    t = np.array([0.5,1.0,1.5,2.0,2.5,3.0])  
    p = np.array([5.0,15.0,20.0,10.0,5.0,0.0])
    data = st.sidebar.data_editor(
        pd.DataFrame([t,p],index=['t [h]','rain [mm]']).T,
        num_rows="dynamic",use_container_width=True)
    t = data['t [h]'].to_numpy()
    p = data['rain [mm]'].to_numpy()
    
    pIntensity = p/(t[1]-t[0])
    
    st.header("")
    st.header("Soil parameters")
    soil = st.sidebar.expander("", expanded=True)
    
    f0 = soil.slider(label='Initial infiltration rate = $f_0$ [mm h$^{-1}$]', 
                   key='f0', value=40.0, min_value=0.0, max_value=200.0)
    fc = soil.slider(label='Final (steady-state) infiltration rate = $f_c$ [mm h$^{-1}$]', 
                   key='fc', value=7.5, min_value=0.0, max_value=100.0)
    k = soil.slider(label='Horton decay constant = $k$ [min$^{-1}$]', 
                  key='k', value=1.0, min_value=0.0, max_value=5.0, step=0.01)


nubersimtimeSteps = 1001; 
simtime = np.linspace(0,max(t),nubersimtimeSteps)
deltaTime = simtime[1]-simtime[0];
f = np.zeros(nubersimtimeSteps,dtype=object)
cumf = np.zeros(nubersimtimeSteps,dtype=object)
intensity = np.zeros(nubersimtimeSteps,dtype=object)
infiltrationRate = np.zeros(nubersimtimeSteps,dtype=object)
infiltration = np.zeros(nubersimtimeSteps,dtype=object)
excess = np.zeros(nubersimtimeSteps,dtype=object)


for tLoop in range(len(t)-1,-1,-1):
    index = simtime<t[tLoop]
    intensity[index] = pIntensity[tLoop]
    
for tLoop in range(len(f)):
    # st.write(tLoop)
    f[tLoop] = fc+(f0-fc)*math.exp(-k*simtime[tLoop])
    cumf[tLoop] = fc*simtime[tLoop]+(f0-fc)*(1.0-math.exp(-k*simtime[tLoop]))/k
    infiltrationRate[tLoop] = min(f[tLoop],intensity[tLoop]);
    excess[tLoop] = max(0,intensity[tLoop]-infiltrationRate[tLoop])

for tLoop in range(len(f)-1):
    infiltration[tLoop+1] = infiltration[tLoop] + \
        0.5*(infiltrationRate[tLoop]+infiltrationRate[tLoop+1])*deltaTime

# Getting values for the excess precipitation above the hyetograph
plotexcess = f+excess

# Figure
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(8,8))
    plt.bar(t-t[0]/2,pIntensity,width=t[1]-t[0])
    plt.plot(simtime,f,'k',label='potential infilration')
    plt.plot(simtime,infiltrationRate,'--r',label='actual infiltration')
    plt.plot(simtime,excess,'--k',label='rainfall excess')    
    plt.fill(np.append(simtime,simtime[::-1]),np.append(plotexcess,f[::-1]),
             'green',label='rainfall-excess')
    plt.xlabel('simtime = t [h]')
    plt.ylabel('infiltration rate = f [mm h$^{-1}$]')
    plt.legend()
    plt.xlim(-0.1,max(t)+0.1)
    st.pyplot(fig)
    
with col2:
    fig = plt.figure(figsize=(8,8))
    plt.plot(simtime,cumf,'k',label='potential infiltration')
    plt.plot(simtime,infiltration,'r',label='Actual infiltration')
    plt.xlabel('simtime = t [h]')
    plt.ylabel('cumulative infiltration = F [mm]')
    plt.legend()
    st.pyplot(fig)
    