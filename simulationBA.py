
#!/usr/bin/python
import argparse
import sys
import json
import ssl
import urllib.request
import numpy as np
import pandas as pd
from csv import reader, writer
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


class Learner(object):

    path = ''
    param = {}
    
    def __init__(self, paramt, loss):
        
        self.param = paramt        
        self.loss = loss
        # self.start_date = paramt['START_DATE_def']
        self.predict_range = paramt['PREDICT_RANGE_def']
        self.s_0 = paramt['S_0_def']
        self.i_0 = paramt['I_0_def']
        self.r_0 = paramt['R_0_def']
        print("todos os dados foram carregados ")

    def load_DaDos(self):
        
        df_data = pd.read_csv(self.param['csv_casosBA'])        
        print(df_data.head())
        df_data = df_data.set_index('Category')
        return df_data

        
    def extend_index(self, index, new_size):
        
        values = index.values
        print("valuessss ", len(values))
        print(index[-1])
        current = datetime.fromisoformat(index[-1])
        
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.fromisoformat(current))
        
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
       
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(
                            modeloSIR, [0, size], [self.s_0, self.i_0, self.r_0], t_eval= np.arange(0, size, 1))

    def train(self):
        #################################################################
        ##    Run the optimization to estimate the beta and gamma    ####
        ##    fitting the given confirmed cases.                     ####
        #################################################################
        
        print('Loading table csv  ....')
        allDados =  self.load_DaDos()

        recovered = allDados['recuperados_BA']        
        death = allDados['Óbitos']        
        confirmados = allDados['Confirmados'] 

        data = (confirmados - recovered - death)
        print("Dados ", data)
        
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
            # 'Susceptible': prediction.y[0],
            'Infected': prediction.y[1], 
            'Recovered': prediction.y[2]            
        }, index= new_index)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Gráficos da Bahia")
        df.plot(ax=ax)
        print(f"beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{'Graficos_COVID_Bahia'}.png")


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
    'S_0_def': 14930634,
    'I_0_def': 2,
    'R_0_def': 10,
    'PREDICT_RANGE_def': 275,
    'START_DATE_def': "2020-03-06",
    'csv_casosBA': "tables/tabelasCasos_Bahia.csv"   
}


download = False
learner = Learner(pmtros, loss)
#try:
learner.train()