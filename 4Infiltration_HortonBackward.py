# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:11:24 2022

@author: Antriksh Srivastava
"""
import math
import streamlit as st
from numpy import arange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.header("Horton's model parameterization")

Documentation = st.expander("Documentation: (Click here to see the documentation)")
with Documentation:
    st.markdown("This page explains how we obtain the horton model's parameters based on double ring infiltrometer \n \
    Using double ring infiltrometer we make sure that the actual measured infiltration is the potential infiltration \n \
    obtained from Horton's model")
    st.subheader("Calculating infiltration rate")
    st.latex(r''' f = \frac{\text{Volume of water added}}{\text{Area*Time}} ''')
    st.latex(r''' f = \frac{V[i]}{A(T(i) - T(i-1))} *1E4 \quad [mm/min]''')
    st.markdown("Where,")
    st.markdown("f = Avereage infiltration rate during this perid [mm/min]")
    st.markdown("AF = After filling [cm]")
    st.markdown("BF = before filling [cm]")
    st.markdown("T = Cumulative time [min]")
    
    st.subheader("Setting up the horton's model")
    st.markdown("Horton's model provides infiltration using an exponential decay function")
    st.latex(r''' f = f_c + (f_o - f_c) \exp ^{-kt} ''')
    st.markdown("Where,")
    st.markdown("$f$ = Actual infiltratoin rate")
    st.markdown("$f_c$ = Steady sate infiltration rate")
    st.markdown("$f_o$ = Infiltration rate just gegining of rainfall")
    st.markdown("$k$ = Horton's decay consant")
    st.markdown("Rearrange the terms we get:")
    st.latex(r''' (f - f_c) = (f_o - f_c) \exp ^{-kt} ''')
    st.latex(r''' \left( \frac{f - f_c}{f_o - f_c} \right) =  \exp ^{-kt} ''')
    st.markdown("Take natural log on LHS & RHS")
    st.latex(r''' ln \left( \frac{f - f_c}{f_o - f_c} \right) =  -kt ''')
    st.latex(r''' ln(f - f_c) - ln(f_o - f_c) =  -kt ''')
    st.latex(r''' ln(f - f_c) = ln(f_o - f_c) -kt ''')
    st.markdown("This seems like a straight line equation")
    st.latex(r''' y = A + Bx ''')
    st.markdown("Where, 'A' & 'B' are constants and are given by :")
    st.latex(r'''A = ln(f_o - f_c)''')
    st.latex(r'''B = -k''')

# Demo data
time = np.array([0,2,5,10,20,30,40,60,80])

beforeFill = np.array([float('nan'),92,93,89,84,89,95,92,91])
afterFill = np.array([100,100,99,101,100,102,101,100,100])
volumefill = np.array([float('nan'),2.26,2.69,3.39,4.52,3.67,1.69,2.54,2.26])

with st.sidebar:
    Area = st.sidebar.number_input("What is the inner ring area [cm$^{2}$]",
                                   key = 'area',value = 2826.0)
    data = st.sidebar.data_editor(pd.DataFrame([time,volumefill],\
                                               index=['Time [min]','Volume added [litre]']).T,
                                               num_rows="dynamic",use_container_width=True)
    t = np.array(data['Time [min]'])
    midtime = (t + np.append(t[1:],np.array([0])))*0.5
    midtime = midtime[:-1]
    V = np.array(data['Volume added [litre]'])
    Compute = st.sidebar.checkbox("Compute parameter",value = False)  
    Manual = st.sidebar.checkbox("Manual",value = False)  
    
diffT = np.diff(t)
diffInfil = V[1:]*1000/Area*10
infilRate = diffInfil/diffT # Observed infiltration rate
fc = infilRate[-1]

st.warning("I assume that the infiltration rate calculated is for \
           middle time point")
st.warning("I assume the last value of the infiltration rate \
           is $f_c$")
           
yy = infilRate - fc  
index = yy > 0
y = np.log(infilRate[index] - fc).reshape(-1,1)
x = midtime[index].reshape(-1,1)
removeind = yy <= 0
if sum(yy < 0):
    st.warning("I am not using some points because they don't \
               make sense to me")               
    
if Compute:
    
    ## Fit the model
    model = LinearRegression()
    # st.write(x,y)
    model.fit(x, y)
    r_sq = model.score(x, y) # R2 for the model
    intercept = model.intercept_ # Intercept of linear model
    coef = model.coef_ # Slope of linear model
    
    ## Computing the parameters
    fo = np.exp(intercept) + fc
    k = -coef
    
    ## Model Calculation
    f = fc + (fo-fc)*np.exp(-k*t)
    f = f.reshape(-1,1)
    yM = fc + (fo-fc)*np.exp(-k*x)
    yM = np.log(yM-fc)
    results1 = " f$_o$ = "+str(np.round(fo[0],2))+ " [mm/min] "
    results2 = " f$_c$ = "+str(np.round(fc,2))+ " [mm/min] "
    results3 = " k = "+str(np.round_(k[0][0],2))+ " [min$^{-1}$] "
    st.success(results1+"  "+results2+"  "+results3)
    
if Manual:
    with st.sidebar:        
        f0M = st.slider(label='Initial infiltration rate = $f_0$ [mm min$^{-1}$]', 
                       key='f0', value=40.0, min_value=0.0, max_value=200.0)
        fcM = st.slider(label='Final infiltration rate = $f_c$ [mm min$^{-1}$]', 
                       key='fc', value=7.5, min_value=0.0, max_value=f0M-1.0)
        kM = st.slider(label='Horton decay constant = $k$ [min$^{-1}$]', 
              key='k', value=1.0, min_value=0.0, max_value=5.0, step=0.01) 
        
    fM = fcM + (f0M-fcM)*np.exp(-kM*t)
    
    fMM = fcM + (f0M-fcM)*np.exp(-kM*x)

    
    index = fMM - fcM > 0
    yMM = np.log(fMM[index] - fcM).reshape(-1,1)
    
    r_sqM = r2_score(yMM,y[index])
    xMM = x[index]
    

    
    
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(8,8))
    plt.scatter(midtime,infilRate,color='b',label='Measured infiltration')
    plt.scatter(midtime[removeind],infilRate[removeind],color='r',\
                    label='Not used in fitting')
    if Compute:
        plt.plot(t,f,color='k',label='Model infiltration')
    if Manual:
        plt.plot(t,fM,color='k',linestyle='dashed',\
                 label='Manual fitting')
        
    plt.legend()
    plt.xlabel('Time [min]')
    plt.ylabel('Infiltration rate [mm/min]')
    plt.title("Actual infiltration")
    st.pyplot(fig)
    
with col2:
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x,y,color='b',label = 'Data')    
    if Compute:
        R2 = f'Optimal: $R^{2}$=[%0.2f]'%r_sq
        plt.plot(x,yM,color='k',label = R2)
    if Manual:
        R2 = f'Manual: $R^{2}$=[%0.2f]'%r_sqM
        plt.plot(xMM,yMM,color='k',linestyle='dashed',label = R2)
        
    plt.title("Model Fitting")
    plt.ylabel('ln($f_o$ - $f_c$)')
    plt.xlabel('Time [min]')
    plt.legend()
    st.pyplot(fig)    
    
# if Start:
#     st.write(data)
#     diffT = np.diff(data['t'])
#     diffD = np.diff(data['D'])
#     infilRate = diffD/diffT
#     st.write(infilRate)
#     t = data.t # min
#     D = data.D # cm
#     f = np.empty(len(t)) # cm/min
#     string = ''
#     f[0] = np.nan

#     n =len(t)
#     for i in range(1,n): 
#         f[i] = ((D[0] - D[i])*60)/(t[i] - t[i - 1]) 
        
#     fc = f[-1]
#     l = np.empty(n)
#     #l[0] = max(math.log(f[i]-fc))
#     P = np.empty(n)
#     P[0]=0 

#     for i in range(1,n):
#         P[i]=f[i]-fc
#         if(P[i]==0):
#             l[i]=0
#         else:    
#             l[i] = math.log(f[i]-fc)
           
#     T = np.empty(n) 

#     for i in range(0,n): 
#         T[i] = t[i]/60
        
#     x = max(l)
#     y = min(l)
#     g = max(T)
#     #plt.plot(T,l)
#     #plt.xlabel('time')
#     #plt.ylabel('ln(f-fc)')
#     #plt.ylim([y,x])
#     #plt.xlim([T[1],g])
#     #plt.legend
#     #plt.plot(T,f)
    
#     #ig, ax = plt.subplots()
#     #ax = plt.plot(T,f, label = 'Infiltration')
#     # Removing the poitns
#     y = np.copy(T)
#     #index = y != 0
#     #y =y[index]
    
#     index = l != 0
#     T = T[index]
#     l = l[index]
#     #index = f != 0
#     #f =f[index]
   
#     def objective(T, a, b):
#      	return a * T + b
            
#     # curve fit
#     popt, _ = curve_fit(objective, T, l)
#     # summarize the parameter values
#     a, b = popt
#     print('y = %.5f * x + %.5f' % (a, b))
#     # plot input vs output
#     #plt.scatter(T, f)
#     # define a sequence of inputs between the smallest and largest known inputs
#     T_line = arange(min(T), max(T), 1)
#     # calculate the output for the range
#     l_line = objective(T_line, a, b)
#     # create a line plot for the mapping function
#     #plt.plot(T_line, l_line, '--', color='red')
#     plt.show()
#     #print(a)    

#     c= math.exp(b) + fc
    
#     F = np.empty(n)
#     for i in range(0,n):
#         F[i] = fc + (c - fc)*math.exp(a*y[i])
    
#     #print(c)
#     st.write("K= ",a)
#     st.write("fo =",c)
#     st.write("fc= ",fc)
    
#     # df['f'] = f
#     # df['fmodel'] = F
#     # fig, ax = plt.subplots()
#     # ax = plt.scatter(y, F)
#     # ax = plt.plot(y,F, label = 'Infiltration')
#     # ax = plt.xlabel('time')
#     # ax = plt.ylabel('F(t)')
#     # ax = plt.legend()
#     # st.pyplot(fig)
#     # st.write(df)