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
    st.header("Calculation of $\phi$-index based on run-off")
    st.latex(r'''  \phi _1 = \text{Initial guess}''')    
    st.markdown("First iteration,")
    st.latex(r'''  \phi _2 = \frac{P_e - R}{t_e}''')
    st.markdown("where,")    
    st.latex(r''' P_e = \text{Effective precipitation [mm] for }t_e ''')
    st.latex(r''' R = \text{Run-ff given [mm]} ''')
    st.latex(r''' t_e = \text{Total duration when }\phi _1 > i \quad [h] \\
             t_e = \Sigma \left( \Delta t (\phi _1 > i) \right)''')
    st.markdown("Further iterations : ")
    st.latex(r'''\text{If : }\phi _2 \quad \neq \quad \phi _1''')
    st.latex(r'''\text{Repeat the steps and get }\phi _3''')
    st.latex(r'''\text{If : }\phi _2 \quad = \quad \phi _1''')
    st.latex(r'''\text{Current value of $\phi$ is the correct value}''')

    
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
    
    st.subheader("Run-off")
    r = st.number_input("Run-off  [mm]",value = 11.0,key = 'runoff')
    
    st.subheader("Intial-guess")
    phiGuess = st.number_input("Intial-$\phi$ assumption [mm /h]",value = 11.0,
                               key='intial',min_value=0.0,max_value=np.max(pIntensity)-2.0)    
        
phi = [phiGuess]
effectiveIndex = phiGuess < pIntensity
tEffective = np.sum(difft[effectiveIndex])
            
infilRate = phiGuess*np.ones([len(pIntensity),])
index = infilRate > pIntensity
infilRate[index] = pIntensity[index]        
infiltration = np.sum(infilRate*difft) 

runoff = np.sum(p) - infiltration       

if tEffective == 0:
    st.error("Please choose a reasonable inidial guess of $\phi$")
else:
    tEffective = [tEffective]
    infiltration = [infiltration]
    runoff = [runoff]
    phiNew = phiGuess + 10 
    intertion = 1
    convergence = False
    while not(convergence):
        effectiveIndex = phi[-1] < pIntensity
        newtEffective = np.sum(difft[effectiveIndex])  
        phiNew = (np.sum(pIntensity[effectiveIndex]*difft[effectiveIndex]) - r)/newtEffective         
                    
        infilRate = phiNew*np.ones([len(pIntensity),])
        index = infilRate > pIntensity
        infilRate[index] = pIntensity[index]        
        infiltrationNew = np.sum(infilRate*difft) 
        
        runoffNew = np.sum(p) - infiltrationNew
        
        if phi[-1] != phiNew:            
            phi.append(phiNew)
            tEffective.append(newtEffective)
            infiltration.append(infiltrationNew)
            runoff.append(runoffNew)
        else:
            convergence = True
        intertion += 1    
    df = pd.DataFrame(np.array([np.array(phi),np.array(tEffective),
                               np.array(infiltration),np.array(runoff)]).T,
                      columns=['phi [mm/h]','te [h]','infiltration [mm]','run-off [mm]'])
    
    
# Figure
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(8,8))
    plt.bar(t-difft/2,pIntensity,width=difft,color='blue',label='Precipitation intensity')
    # plt.bar(t-difft/2,phiarray,width=difft,color='green',label='Infiltration intensity')
    
    for i in range(len(phi)):
        string = 'Interation-'+str(i+1)
        if i == len(phi)-1:            
            plt.axhline(y = phi[i], color = 'r', linestyle = '-',label=string)
        else:
            plt.axhline(y = phi[i], color = 'k', linestyle = '-',label=string)
    plt.xlabel('time  [h]')
    plt.ylabel('intensity  [mm h$^{-1}$]')
    plt.legend()
    plt.xlim(-0.1,max(t)+0.1)
    st.pyplot(fig)
    
with col2:
    st.subheader("Iteration steps")
    st.dataframe(df,use_container_width=True)
   