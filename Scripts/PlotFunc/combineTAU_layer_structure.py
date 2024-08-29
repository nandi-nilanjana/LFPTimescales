#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:21:28 2024

@author: nilanjana
"""

#each session folder in '/Volumes/work/comco/nandi.n/LFP_timescales/Results/FOOOF' has layerwise_fooof_results 
#each pickle file ahs layer info, structure, knee frquency and timescale value (in ms)

#run the loop of sessons
 
#for each loop load the pickle file 

#only check the timescale value if the knee freq is between 1 and 150 since i computed the psd computation in this freq range 

#create one dataframe with all the following headers  - monkey, layer , structure , tau 

#i can always take the specific info from this dataframe later to do the histogram plot or do the anov analysis

#import modules
import os
import pickle 
import re
import numpy as np 
import pandas as pd
import netCDF4 as nc

from scipy.stats import kruskal
#session names for which the fooof results were extracted (30 sessions in Mourad, sometimes a single session has 2 probes in PMd)
#need to extract and add the single M1 session in mourad 
sessions = ['Mo180405001','Mo180405004',
            'Mo180418002','Mo180419003','Mo180426004','Mo180503002', 'Mo180523002','Mo180524003', 
            'Mo180525003','Mo180531002','Mo180614002','Mo180614006','Mo180615002','Mo180615005', 'Mo180619002',
            'Mo180620004','Mo180622002','Mo180626003', 'Mo180627003','Mo180629005', 'Mo180703003','Mo180704003', 
            'Mo180705002','Mo180706002', 'Mo180710002','t140924003','t140925001',
            't140926002','t140929003','t140930001','t141001001','t141008001','t141010003','t150122001','t150123001',
            't150128001','t150204001','t150205004','t150212001','t150303002','t150319003','t150327002','t150327003',
            't150415002','t150416002','t150423002','t150430002','t150520003','t150716001']


#s =['Mo180328001','Mo180711004','Mo180712006']
# sessions = ['t140924003','t140925001',
# 't140926002','t140929003','t140930001','t141001001','t141008001','t141010003','t150122001','t150123001',
# 't150128001','t150204001','t150205004','t150212001','t150303002','t150319003','t150327002','t150327003',
# 't150415002','t150416002','t150423002','t150430002','t150520003','t150716001']
# #add Tomys files

current_path = os.getcwd()
if current_path.startswith('/Users'):
    server = '/Volumes/work'  # local w VPN
elif current_path.startswith('/home/'):
    server = '/envau/work/'  # niolon
elif current_path.startswith('/envau'):
    server = '/envau/work/'  # niolon

freq_thres = [2,80] #frequencies over whihc the psd were estimated in fooof

#%%
all_data = []
for s, sess in enumerate(sessions):
    
    #path of pickle file
    f_path = f'{server}/comco/nandi.n/LFP_timescales/Results/FOOOF/{sess}/layerwise_fooof_results.pkl'
    
    #load file 
    with open (f_path,'rb') as file:
        data = pickle.load(file)
    print (f'Processing session {sess}')
    for f, freq in enumerate(data['knee']):
        #print(freq)
        if freq >= freq_thres[0] and freq <= freq_thres[1]:
            print(freq)
            t = (data['tau'][f]) #timescale
            l = (data['layer'][f]) #layer info
            struc = data['area'][f] #structure M1/PMd
            
            if struc == 'premotor':
                struc = 'PMd' #for the sessions with two probes in PMd , for extracted they were deisgnated as PMd and premotor
            
            if  data['session'][f].startswith('Mo'):
                
                m_name =  'Mourad'
            else:
                m_name =  'Tomy'
                
                
            n_data = {'monkey': m_name,
                      'tau' : t,
                      'area' : struc,
                      'layer' : l
                      }
            all_data.append(n_data) #keep appending to the list
        else:
            a =  data['area'][f]
            l = data['layer'][f]
            print(f'Skipping Session {sess} - {a} - {l}')

df = pd.DataFrame(all_data)

outFolder = f'{server}/comco/nandi.n/LFP_timescales/Results/timescale_estimate'
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

f_save = 'fooof_results_combined_layerAveraged.pkl'    


#save the datafrmae in the pickle file
with open (os.path.join(outFolder,f_save), 'wb') as outfile:
    pickle.dump(df,outfile)
 
    
#%% load the the individual channels and their tau values       

all_data = []
freq_thres = [0,0] 
c=0
keep_c = 0 

for s, sess in enumerate(sessions):
    #path of .nc file
    inFolder = f'{server}/comco/nandi.n/LFP_timescales/Results/FOOOF/fixed/{sess}' 

    for f in os.listdir(inFolder):
        if f.endswith('.nc'):
            
            f_path = os.path.join(inFolder,f)
            ds = nc.Dataset(f_path)
            area = re.split(r'_', f)[1] #take from the file name
            #dict keys - ds.variables.keys()
            knee = ds['laminar_knees']
            layers = ds.variables['layers']
            
            if layers.shape[0] == knee.shape[0]:
            
                if area == 'premotor':
                    area = 'PMd' #for the sessions with two probes in PMd , for extracted they were deisgnated as PMd and premotor
           
                elif area == 'motor':
                    area = 'M1'
                    
                if f.startswith('Mo'):
                    m_name =  'Mo'
                else:
                    m_name =  'Tomy'
                    
                if 'laminar_tau_ms' in ds.variables: 
                    #where 1/f was computed in knee mode and tau is computed from knee frequency
                    tau=ds.variables['laminar_tau_ms']
                   
                    for i in range (layers.shape[0]): 
                        c+=1 #counter to check how many channels are there initially irrespective of good knee
                        
                        if knee[i].data >= freq_thres[0] and knee[i].data <= freq_thres[1]:
                           n_data = {'session': sess,
                                      'tau' : tau[i].data,
                                      'area' : area,
                                      'layer' : layers[i] ,   
                                       'knee' : knee[i].data, 
                                       'monkey' : m_name
                                      }
                            
                           all_data.append(n_data) #keep appending to the list
                           keep_c+=1
                else:
                    # only for those files where 1/f exp was calculated without knee that is in fixed mode
                    
                    for i in range (layers.shape[0]): 
                        c+=1 #counter to check how many channels are there initially irrespective of good knee
                        
                        
                        n_data = {'session': sess,
                                   
                                   'area' : area,
                                   'layer' : layers[i] ,   
                                    'knee' : knee[i].data,#here knee means basically the mono exponent, not real knee
                                    'monkey' : m_name
                                   }
                        
                        all_data.append(n_data) #keep appending to the list
                          
                    
                    
            else:
                print(f'Error in file: layer and tau length do not match! {f}')
                
            
           
                        
df = pd.DataFrame(all_data)
 

outFolder = f'{server}/comco/nandi.n/LFP_timescales/Results/timescale_estimate'
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

f_save = 'fooof_results_combined_allChannels.pkl'

# #save the datafrmae in the pickle file
# with open (os.path.join(outFolder,f_save), 'wb') as outfile:
#     pickle.dump(df,outfile)    
    
    
#%% kruskal wallis and post hoc analysis to compare
#if there is any main effect of monkey, area and layer on tau values 

import pandas as pd
from scipy.stats import kruskal

# Function to perform Kruskal-Wallis test
def kruskal_test(df, factor, response):
    groups = df.groupby(factor)[response].apply(list).values
    kruskal_stat, kruskal_p = kruskal(*groups)
    return kruskal_stat, kruskal_p

# Perform Kruskal-Wallis test for each factor (Find main effect of monkey, area or layer on tau values)
factors = ['monkey', 'area', 'layer']
#factors = ['area','layer']
response = 'knee'

for factor in factors:
    kruskal_stat, kruskal_p = kruskal_test(df, factor, response)
    print(f"Kruskal-Wallis Test for {factor} effect on {response}:")
    print(f"Statistic: {kruskal_stat}, P-value: {kruskal_p}\n")


# 1. Analyze the effect of Area on TAU for each Monkey (Layer Collapsed)
monkeys = df['monkey'].unique()
for monkey in monkeys:
    subset = df[df['monkey'] == monkey]
    kruskal_stat, kruskal_p = kruskal_test(subset, 'area', response)
    print(f"Kruskal-Wallis Test for Area effect on TAU within Monkey = {monkey}:")
    print(f"Statistic: {kruskal_stat}, P-value: {kruskal_p}\n")
   

# 2. Analyze the effect of Layer on TAU for each Monkey (Area collapsed)
for monkey in monkeys:
    subset = df[df['monkey'] == monkey]
    areas = subset['area'].unique()
    for area in areas:
        area_subset = subset[subset['area'] == area]
        kruskal_stat, kruskal_p = kruskal_test(area_subset, 'layer', response)
        print(f"Kruskal-Wallis Test for Layer effect on TAU within Monkey = {monkey} and Area = {area}:")
        print(f"Statistic: {kruskal_stat}, P-value: {kruskal_p}\n")
        
        

# t_PMd = df[(df['monkey'] == 'Tomy') & (df['area'] == 'PMd')]
# t_M1 = df[(df['monkey'] == 'Tomy') & (df['area'] == 'M1')]
# np.median(t_PMd['tau'].values)
# print(f'PMd median: {np.median(t_PMd['tau'].values)}')
# print(f'PMd median: {np.median(t_M1['tau'].values)}')

#a lot of channels are anyway discarded since they dont have knee

# 11 session in Tomy PMd where knee was found 
# 9 sessions in Tomy M1 where knee was found 


#%%
'''
Trying to figure out if i exclude knee, then what slope can i use for fitting. I need to start the fits from a frequency range
not affected by knee, since some channels and sessions have knee . 

'''
#find the max knee value in all channels irresptive of monkey , area or layer
m = df['knee'].max()

#find the session and and the channels with this max knee 

df_m = df[df['knee']==m]

freq_th = 35

df_th = df[df['knee']>freq_th]

#%%
'''
Individual comparison using kruskal wallis between layers and between areas 
'''
p_sig = 0.05
monkey = ['Tomy', 'Mo']

#Mourad

#area all layers


for i, m in enumerate(monkey):
    
    exp_PMd = df[(df['monkey'] == m) & (df['area'] == 'PMd')]['knee']
    exp_M1 = df[(df['monkey'] == m) & (df['area'] == 'M1')]['knee']
    kruskal_s, kruskal_pval = kruskal(exp_PMd,exp_M1)
    
    if kruskal_pval < p_sig:
        print(f"Kruskal-Wallis Test for area effect on exp within Monkey = {m} and median_M1 = {np.round(np.median(exp_M1),2)} and median_PMd ={np.round(np.median(exp_PMd),2)}")
    else:
        print(f'No significant difference between M1/PMd in {m}')


#layers

area = ['M1', 'PMd']

for j, a in enumerate(area):
    for m in monkey:
        
        exp_L23 = df[(df['monkey'] == m) & (df['area'] == a) & (df['layer']== 'L23')]['knee']
        exp_L5 = df[(df['monkey'] == m) & (df['area'] == a) & (df['layer']== 'L5')]['knee']
        exp_L6= df[(df['monkey'] == m) & (df['area'] == a) & (df['layer']== 'L6')]['knee']
        
        _, p_L23 = kruskal(exp_L23,exp_L5)
        _, p_L5 = kruskal(exp_L5,exp_L6)
        _, p_L6 = kruskal(exp_L6,exp_L23)
        
        if p_L23 < p_sig:
            print(f'Significant diff between L23 (Median = {np.round(np.median(exp_L23),2)}) and L5 (Median = {np.round(np.median(exp_L5),2)}) in area = {a} and Monkey = {m} with p = {p_L23}')
        
        if p_L5 <p_sig:
            print(f'Significant diff between L5 (Median = {np.round(np.median(exp_L5),2)} and L6 (Median = {np.round(np.median(exp_L6),2)}) in area = {a} and Monkey = {m} with p = {p_L5}')
        
        if p_L6< p_sig:
            print(f'Significant diff between L6 (Median = {np.round(np.median(exp_L6),2)}) and L23 (Median = {np.round(np.median(exp_L23),2)}) in area = {a} and Monkey = {m} with p = {p_L6}')
            
           
    
        
        
    













