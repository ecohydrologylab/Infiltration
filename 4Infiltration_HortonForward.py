# Horton's method forward

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.header("Horton's method")
st.latex(r'''  f = f_c + (f_0-f_c)\exp(-kt) ''')
st.latex(r'''  F = f_c t + (f_0-f_c)\left[\frac{1-\exp(-kt)}{k}\right] ''')
st.markdown("Where,")
st.markdown("$f$ = Actual infiltration rate &emsp; [mm min$^{-1}$]")
st.markdown("$f_c$ = Steady sate infiltration rate &emsp; [mm min$^{-1}$]")
st.markdown("$f_o$ = Infiltration rate just gegining of rainfall &emsp; [mm min$^{-1}$]")
st.markdown("$k$ = Horton's decay consant &emsp; [min$^{-1}$]")
st.markdown("$F$ = Potential cumulative Infiltration &emsp; [mm]")

# Add axis lables for the figure
# If we can make the plots with plotly it will be easy to get the values

# parameters = st.sidebar(label='Horton Parameters')
with st.sidebar:
    
    st.header("Horton's parameters")
    
    t = st.number_input(label='Total simulation time = $t$ [min]', key='t', 
                    value=100.0)
    
    st.header("")
    st.header("Soil 1")
    soil1 = st.sidebar.expander("", expanded=True)
    
    f01 = soil1.slider(label='Initial infiltration rate = $f_0$ [mm min$^{-1}$]', 
                   key='f01', value=50.0, min_value=0.0, max_value=200.0)
    fc1 = soil1.slider(label='Final infiltration rate = $f_c$ [mm min$^{-1}$]', 
                   key='fc1', value=10.0, min_value=0.0, max_value=100.0)
    k1 = soil1.slider(label='Horton decay constant = $k$ [min$^{-1}$]', 
                  key='k1', value=0.1, min_value=0.0, max_value=2.0, step=0.01)
    
    st.header("")
    st.header("Soil 2")
    soil2 = st.sidebar.expander("", expanded=True)
    
    f02 = soil2.slider(label='Initial infiltration rate = $f_0$ [mm min$^{-1}$]', 
               key='f02', value=50.0, min_value=0.0, max_value=200.0)
    fc2 = soil2.slider(label='Final infiltration rate = $f_c$ [mm min$^{-1}$]', 
               key='fc2', value=10.0, min_value=0.0, max_value=100.0)
    k2 = soil2.slider(label='Horton decay constant = $k$ [min$^{-1}$]', 
              key='k2', value=0.1, min_value=0.0, max_value=2.0, step=0.01)


tSteps = np.zeros(101,dtype=object)
f1 = np.zeros(101,dtype=object)
cumf1 = np.zeros(101,dtype=object)
f2 = np.zeros(101,dtype=object)
cumf2 = np.zeros(101,dtype=object)
for tLoop in range(len(f1)):
    tSteps[tLoop] = tLoop*t/100
    f1[tLoop] = fc1+(f01-fc1)*math.exp(-k1*tSteps[tLoop])
    cumf1[tLoop] = fc1*tSteps[tLoop]+(f01-fc1)*(1.0-math.exp(-k1*tSteps[tLoop]))/k1
    f2[tLoop] = fc2+(f02-fc2)*math.exp(-k2*tSteps[tLoop])
    cumf2[tLoop] = fc2*tSteps[tLoop]+(f02-fc2)*(1.0-math.exp(-k2*tSteps[tLoop]))/k2
    
# Figure
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(8,8))
    plt.plot(tSteps,f1,label='Soil 1')
    plt.plot(tSteps,f2,label='Soil 2')
    plt.xlabel('time = t [min]')
    plt.ylabel('infiltration rate = f [mm m$^{-1}$]')
    plt.legend()
    st.pyplot(fig)
with col2:
    fig = plt.figure(figsize=(8,8))
    plt.plot(tSteps,cumf1,label='Soil 1')
    plt.plot(tSteps,cumf2,label='Soil 2')
    plt.xlabel('time = t [min]')
    plt.ylabel('cumulative infiltration = F [mm]')
    plt.legend()
    st.pyplot(fig)