import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.optimize import Bounds
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


#!/usr/bin/python
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request



class Learner(object):

    path = ''
    param = {}
    
    def __init__(self, paramt, loss, mPais):
        
        self.param = paramt
        self.country = mPais
        self.loss = loss
        self.start_date = paramt['START_DATE_def']
        self.predict_range = paramt['PREDICT_RANGE_def']
        self.s_0 = paramt['S_0_def']
        self.i_0 = paramt['I_0_def']
        self.r_0 = paramt['R_0_def']
        print("todos os dados foram carregados ")

    def load_confirmed(self):
        
        df = pd.read_csv(self.param['csv_confirmed'])
        country_df = df[df['Country/Region'] == self.country]
        print(country_df.head())
        serie = country_df.iloc[0].loc[self.start_date:]
        print(serie)
        return serie

    def load_recovered(self):
        
        df = pd.read_csv(self.param['csv_recovered'])
        print("show 5 files of " + self.country)
        
        country_df = df[df['Country/Region'] == self.country]
        print(country_df.head())      
        serie = country_df.iloc[0].loc[self.start_date:]  
        print(serie)
        return serie

    def load_dead(self):
        
        df = pd.read_csv(self.param['csv_death'])
        country_df = df[df['Country/Region'] == self.country]
        print(country_df.head())
        serie = country_df.iloc[0].loc[self.start_date:]
        print(serie)
        return serie
    
    def extend_index(self, index, new_size):
        
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        
        return values

    def predict(self, beta, gamma, data, recovered, death):
        #beta, gamma, data, , country, s_0, i_0, r_0
        """
            Predict how the number of people in each compartment can be 
            changed through time toward the future.
            The model is formulated with the given beta and gamma.
        """
        predict_range = 150
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        
        def modeloSIR(t, y):  # t, 
            
            S = y[0]
            I = y[1]
            R = y[2]
            
            #dS/dt = -BIS 
            dS_dt = -beta * S * I
            # dR/dt = gI
            dR_dt = gamma * I
            # dI/dt = BIS -gI
            dI_dt = (beta * I * S) - dR_dt           
            
            return [dS_dt, dI_dt, dI_dt]
        
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
       
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(modeloSIR, [0, size], [self.s_0, self.i_0, self.r_0], t_eval= np.arange(0, size, 1))

    def train(self):
        """
            Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        print('Loading table csv RECOVERED ....')
        recovered = self.load_recovered()
        
        print('Loading table csv DEAD ....')
        death = self.load_dead()
        
        print('Loading table csv CONFIRMED ....')
        data = (self.load_confirmed() - recovered - death)
        
        bounds = Bounds([0.00000001, 0.4], [0.00000001, 0.4])
        
        optimal = minimize(
            loss,
            [0.001, 0.001],
            args=(data, recovered, self.s_0, self.i_0, self.r_0),
            method='L-BFGS-B',
            bounds=bounds
        )
        print("modelo optimal \n", optimal)
        
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data, recovered, death)
        
        df = pd.DataFrame({
            'Infected data': extended_actual,
            'Recovered data': extended_recovered,
            'Death data': extended_death,
            'Susceptible': prediction.y[0],
            'Infected': prediction.y[1], 
            'Recovered': prediction.y[2]            
        }, index= new_index)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.country}.png")


def loss(point, data, recovered, s_0, i_0, r_0):
        
    # RMSE between actual confirmed cases and the estimated infectious 
    # people with given beta and gamma.
    
    size = len(data)
    beta, gamma = point
    
    def SIR(t, y):
        
        S = y[0]
        I = y[1]
        R = y[2]

        #dS/dt = -BIS 
        dS_dt = -beta * S * I
        # dR/dt = gI
        dR_dt = gamma * I
        # dI/dt = BIS -gI
        dI_dt = (beta * I * S) - dR_dt           
        
        return [dS_dt, dI_dt, dI_dt]

    solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2


pmtros = {
    'S_0_def': 100000,
    'I_0_def': 2,
    'R_0_def': 10,
    'PREDICT_RANGE_def': 150,
    'START_DATE_def': "1/22/20",
    'COUNTRY_CSV_def': ['Brazil'],
    'csv_confirmed': 'tables/time_series_covid19_confirmed_global.csv',
    'csv_death': 'tables/time_series_covid19_deaths_global.csv',
    'csv_recovered': 'tables/time_series_covid19_recovered_global.csv'
}

# r_0 = pmtros['r_0_def']
# i_0 = pmtros['I_0_def']
# s_0 = pmtros['S_0_def']
# predict_range = pmtros['PREDICT_RANGE_def']
# startdate = pmtros['START_DATE_def']
# countries = pmtros['COUNTRY_CSV_def']
download = False

for pais in pmtros['COUNTRY_CSV_def']:
    
    learner = Learner(pmtros, loss, pais)
    #try:
    learner.train()
    #except BaseException:
    #    print('WARNING: Problem processing ' + str(country) +
    #        '. Be sure it exists in the data exactly as you entry it.' +
    #        ' Also check date format if you passed it as parameter.')
        

