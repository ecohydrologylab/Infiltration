# Horton's method forward

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True)

documentation = st.expander("Documentation for $\phi$ index method")
with documentation:
    st.header("Infiltration and rainfall excess calculation")
    st.latex(r'''  f = min(P,\phi) ''')
    # st.latex(r'''  F = f_c t + (f_o-f_c)\left[\frac{1-\exp(-kt)}{k}\right] ''')
    st.markdown("Where,")
    st.markdown("$P$ = Precipitation intesity [mm h$^{-1}$]")
    st.markdown("$\phi$ = Constant infiltration rate; [mm h$^{-1}$]")
    st.markdown("$f$ = Actual infltration rate &emsp; [mm h$^{-1}$]")
    st.latex(r'''  F = \large{\Sigma \left( i \Delta t \right)} ''')

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
        
    
    difft = np.diff(t)
    difft = np.insert(difft,0,np.array([t[0]]))
    pIntensity = p/difft
    
    
    st.header("")
    st.header("Soil parameters")
    soil = st.sidebar.expander("", expanded=True)
    
    phi = soil.slider(label='Constant infiltraion rate =  $\phi$ [mm h$^{-1}$]', 
                   key='phi', value=11.0, min_value=0.0, max_value=200.0)
        
    phiarray = phi*np.ones([len(pIntensity),])
    index = phiarray > pIntensity
    phiarray[index] = pIntensity[index]
    cumf = np.cumsum(phiarray*difft)    
    cumf = np.insert(cumf,0,0)
    potf = np.cumsum(phi*difft)
    potf = np.insert(potf,0,0)
    simtime = np.insert(t,0,0)
        
    
# Figure
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(8,8))
    plt.bar(t-difft/2,pIntensity,width=difft,color='blue',label='Precipitation intensity')
    plt.bar(t-difft/2,phiarray,width=difft,color='green',label='Infiltration intensity')
    plt.axhline(y = phi, color = 'k', linestyle = '-')
    plt.xlabel('time  [h]')
    plt.ylabel('intensity  [mm h$^{-1}$]')
    plt.legend()
    plt.xlim(-0.1,max(t)+0.1)
    st.pyplot(fig)
    
with col2:
    fig = plt.figure(figsize=(8,8))
    plt.plot(simtime,potf,'k',label='Potential infiltration')
    plt.plot(simtime,cumf,'r',label='Actual infiltration')
    plt.xlabel('time  [h]')
    plt.ylabel('cum. Infiltration  [mm]')
    plt.legend()
    st.pyplot(fig)