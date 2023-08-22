# Horton's method forward

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.header("SCS curve number method for rainfall event")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Make the table in the center
# Better to use st.dataframe (check how to make it dynamic?)
# Make one big colum for A/B/C/D and call it HSG and then 2 subcolumns for area and CN
# Add theory and equations for SCS curve number and AMC conditions
# Display only 2 significant digits for weighted CN

landUse = pd.array(['Soil1','Soil2'],dtype = "string")
aArea = np.array([30,2])
aCN = np.array([80.0,85.0])

bArea = np.array([22,12])
bCN = np.array([80.0,85.0])

cArea = np.array([11,5])
cCN = np.array([80.0,85.0])

dArea = np.array([9,9])
dCN = np.array([80.0,85.0])
        
df = pd.DataFrame(np.array([landUse,aArea,aCN,bArea,bCN,cArea,cCN,dArea,dCN]).T,
                  columns=['Land Use','A %','A CN','B %','B CN',
                             'C %','C CN','D %','D CN'])

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
input {
    font-size: 2rem !important;
}
</style>
""", unsafe_allow_html=True)


if 'df' not in st.session_state:
    st.session_state.df = df
    
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
          

col1, col2 = st.columns([1.2,1])

with col1:
    doc = st.expander("Theory:")
    with doc:
        st.markdown('<p class="big-font">Curve number of a combination of land use land cover</p>', unsafe_allow_html=True)
        st.latex(r'''CN(II)_{wt} = \frac{\sum_{i=1}^{n} CN(II)_i w_i }{ \sum_{i=1}^{n} w_i}''')        
        st.markdown("where,")
        st.markdown("$CN(II)_{wt}$ = Weighted CN for the water shed")
        st.markdown("$CN(II)_{i}$ = Curve number of i$^{th}$ soil group")
        st.markdown("$w_{i}$ = weightage fraction of i$^{th}$ soil group")
        st.markdown('<p class="big-font">Curve Number needs to be modified based on the antecendent conditions:</p>', unsafe_allow_html=True)
        st.markdown("$i)$ Dry conditions $(AMC: I)$") 
        st.latex(r'''CN(I) = \frac{4.2 CN(II)}{10 - 0.058 CN(II)}''')
        st.markdown("$ii)$ Normal antecedent moisture conditions $(AMC: II)$")
        st.markdown("$iii)$ Wet conditions $(AMC: III)$")
        st.latex(r'''CN(III) = \frac{23 CN(II)}{10 + 0.13 CN(II)}''')
        st.markdown('<p class="big-font">Calculating retention/storage from curve number:</p>', unsafe_allow_html=True)
        st.latex(r''' S = \frac{1000}{CN} - 10 \quad [in] ''')
        st.markdown("where,")        
        st.latex(r'''S = \text{Potential maximum retention [in]} \\
                 CN = \text{Curve Number}''')
        st.markdown('<p class="big-font">Calculating effective precipitation:</p>', unsafe_allow_html=True)         
        st.latex(r'''P_e = \frac{(P - 0.2S)^{2}}{(P + 0.8S))}''')
        st.markdown("where")
        st.latex(r'''P_e = \text{Rainfall excess }\quad [in] \\
                 P = \text{Tatal rainfall }\quad [in]''')                 
                 
with col2: 
    st.subheader("Detail about the land use cover")                
    df = st.data_editor(df,num_rows="dynamic",use_container_width=True)    
    landUse = df['Land Use'].to_numpy()
    aArea= df['A %'].to_numpy()
    aCN = df['A CN'].to_numpy()
    bArea = df['B %'].to_numpy()
    bCN = df['B CN'].to_numpy()
    cArea = df['C %'].to_numpy()
    cCN = df['C CN'].to_numpy()
    dArea = df['D %'].to_numpy()
    dCN = df['D CN'].to_numpy()  
    
if (aArea[-1] is None) or (bArea[-1] is None) or (cArea[-1] is None) or (dArea[-1] is None) or \
    (aCN[-1] is None) or (bCN[-1] is None) or (cCN[-1] is None) or (dCN[-1] is None) or \
        (landUse[-1] is None):
    st.warning("Fill all the entries first")    
else:
    with col2:    
        if sum(aArea+bArea+cArea+dArea) == 100:
            flag = False
            expandHSG = st.expander("Hydrologic Soil Group: Division")
            with expandHSG:
                # Creating dataset
                labels = ['HSG: A', 'HSG: B', 'HSG: C','HSG: D']         
                data = np.round_([aArea.sum(),bArea.sum(),cArea.sum(),dArea.sum()],2)  
                fig = go.Figure(
                go.Pie(
                labels = labels,
                values = data,
                hoverinfo = "label+percent",
                textinfo = "value"
                ))
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                  width=550,height=500)        
                st.header("Hydrologic Soil Group Division")
                st.plotly_chart(fig)
            expanderEachHSG = st.expander("Individual HSG Division")
            with expanderEachHSG:
                col21,col22 = st.columns(2)
                with col21:
                    # Hydrologic Soil group A dicision
                    data = aArea/aArea.sum()*100
                    fig = go.Figure(go.Pie(
                    labels = landUse,
                    values = data,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                      width=250,height=200)
                    st.header("HSG: A division")
                    st.plotly_chart(fig)
                    
                    # Hydrologic Soil group B dicision
                    data = bArea/bArea.sum()*100
                    fig = go.Figure(go.Pie(
                    labels = landUse,
                    values = data,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                      width=250,height=200)
                    st.header("HSG: B division")
                    st.plotly_chart(fig)
                with col22:
                    # Hydrologic Soil group C dicision
                    data = cArea/cArea.sum()*100
                    fig = go.Figure(go.Pie(
                    labels = landUse,
                    values = data,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                      width=250,height=200)
                    st.header("HSG: C division")
                    st.plotly_chart(fig)
                    
                    # Hydrologic Soil group D dicision
                    data = dArea/dArea.sum()*100
                    fig = go.Figure(go.Pie(
                    labels = landUse,
                    values = data,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                                      width=250,height=200)
                    st.header("HSG: D division")
                    st.plotly_chart(fig)
    if sum(aArea+bArea+cArea+dArea) == 100:
        with col1:                            
            flag = False             
            CN2 = sum((aArea*aCN+bArea*bCN+cArea*cCN+dArea*dCN))/100.0
            S2 = 1000/CN2 - 10
            CN1 = (4.2*CN2)/(10 - (0.058*CN2))
            S1 = 1000/CN1 - 10
            CN3 = (23*CN2)/(10 + (0.13*CN2))
            S3 = 1000/CN3 - 10        

        with col2:    
            col21,col22 = st.columns(2)    
            with col21:
                st.subheader("Curve number calculation")
                st.dataframe(pd.DataFrame({'CN(I)':[CN1,S1,0.2*S1],
                                           'CN(II)':[CN2,S2,0.2*S2],
                                           'CN(III)':[CN3,S3,0.2*S3]},\
                                      index=["CN","S [in]","Ia [in]"]),use_container_width=True)
            with col22:
                st.subheader("Choose the curve number")
                curveNumber = st.radio(label=" ",index=1,
                                       options=['CN(I)','CN(II)','CN(III)'],horizontal=True)
                
            if curveNumber == 'CN(II)':
                S = 25.4*S2*np.ones([len(pIntensity),])
            elif curveNumber == 'CN(I)':
                S = 25.4*S1*np.ones([len(pIntensity),])
            elif curveNumber == 'CN(III)':
                S = 25.4*S3*np.ones([len(pIntensity),])
                
            Ia = 0.2*S
            P = np.cumsum(p)
            abstractionIndex = Ia > P
            Ia[abstractionIndex] = P[abstractionIndex] 
            Fa = S*(P-Ia)/(P-Ia+S)
            Pe = P - Ia - Fa
            PeIntensity = np.diff(Pe)
            PeIntensity = np.insert(PeIntensity,0,Pe[0])/difft
            df2 = pd.DataFrame(np.array([t,P,Ia,Fa,Pe,PeIntensity]).T,columns=
                               ["t [h]","cum. P [mm]","cum. Ia [mm]","cum. Fa [mm]",
                                "cum. Pe [mm]","Pe intensity [mm/h]"])
            st.dataframe(df2)
           
        
        with col1:
            fig = plt.figure(figsize=(8,8))
            plt.bar(t-difft/2,pIntensity,width=difft,color='green',label='Excess precipitation')        
            plt.bar(t-difft/2,pIntensity-PeIntensity,width=difft,color='blue',label='Infiltration SCS rate') 
            plt.xlabel('time  [h]')
            plt.ylabel('intensity  [mm h$^{-1}$]')
            plt.legend()
            plt.xlim(-0.1,max(t)+0.1)
            st.pyplot(fig)
    else:
        flag = True
        st.error('Error: Sum of area classes should be 100%')             