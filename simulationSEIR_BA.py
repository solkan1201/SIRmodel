import pandas as pd
import numpy as np
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import odeint

############################################################################
###########        Definiendo a equação diferencial       ##################
####   todas as variaveis deveram ser atualizadas a 1                #######
####  incluir a primeira entrada em branco no vetor para beta, gama, p #####
####  para que os índices se alinhem nas equações e no código.         #####
#### No futuro, pode incluir recuperação ou infecção da classe exposta #####
#### (assintomáticos)                                                  #####

def modeloSEIR(Ii, t, beta, taxa_prog, gamma, taxa_infec, taxa_obito, N): 
    
    # Individuos suceptiveis    
    Sscept= N - sum(Ii)
    # individuos expostos [infectados, mas ainda não infecciosos ou sintomáticos] 
    indExp= np.dot(beta[1:3], Ii[1:3]) * Sscept - taxa_prog * Ii[0]

    ### Infecção leve (não é necessária hospitalização)
    inf_leve= (taxa_prog * Ii[0]) - (gamma[1] + taxa_infec[1]) * Ii[1] #I1
    
    ### Infecção grave (hospitalização necessária)
    inf_grave= (taxa_infec[1] * Ii[1]) - (gamma[2] + taxa_infec[2]) * Ii[2] #I2
    
    ### Infecção crítica (UTI necessária)
    inf_crit= (taxa_infec[2] * Ii[2]) - (gamma[3] + taxa_obito) * Ii[3] #I3

    ### indivíduos que se recuperaram da doença 
    recup= np.dot(gamma[1:3], Ii[1:3]) #R
    ### Dead individuals
    obitos= taxa_obito * Ii[3] #D

    return [indExp, inf_leve, inf_grave, inf_crit, recup, obitos]

def calculo_taxa_repR0(pmts):
    
    Npop = pmts['sizePop']
    beta = np.zeros(4)
    gamma = np.zeros(4)
    taxaInf = np.zeros(3)

    # calculando parametros necesarios 
    tax_prog = 1 / pmts['per_incub']    # alpha
    
    valorTemp = 1 / pmts['t_intern_obit']
    tax_Ob = valorTemp * (pmts['taxa_obit'] / pmts['frac_inf_cri'])   # u  miu 
    gamma[3] = valorTemp - tax_Ob

    valorTemp = 1 / pmts['dur_hosp']
    taxaInf[2] =  valorTemp * ( pmts['frac_inf_cri'] / ( pmts['frac_inf_cri'] +  pmts['frac_inf_sev']))
    gamma[2] = valorTemp - taxaInf[2]

    valorTemp = 1 / pmts['dur_infect_leve']
    gamma[1] = valorTemp * pmts['frac_inf_leve']
    taxaInf[1] =  valorTemp - gamma[1]

    #beta= 2e-4 * np.ones(4) # todos os estágios transmitem igualmente
    beta = 2.5e-4 * np.array([0,1,0,0]) # casos hospitalizados não transmitem

    # Calculo do radio de infeção basico 
    R0 = Npop * ((beta[1] / (taxaInf[1] + gamma[1])) + 
                (taxaInf[1] / (taxaInf[1] + gamma[1]))*(
                    beta[2] / (taxaInf[2] + gamma[2]) + 
                    (taxaInf[2] / (taxaInf[2] + gamma[2])) * (beta[3] / (tax_Ob + gamma[3]))))
    
    
    
    return R0, Npop, tax_prog, beta, gamma, taxaInf, tax_Ob


def trainingModelSIER (pmts):
    
    tmax= 365
    tvec= np.arange(0,tmax,0.1)
    ic = np.zeros(6)
    ic[0]= 1
        
    R0, NNpop, atax_prog, bbeta, ggamma, ptaxaInf, utax_Ob = calculo_taxa_repR0(pmts)
    print("O radio de infeão basico : {0:4.3f}". format(R0))
    print("população da Bahia: {}".format(NNpop))

    soln = odeint(modeloSEIR, ic, tvec, args=(bbeta, atax_prog, ggamma, ptaxaInf, utax_Ob, NNpop))
    soln = np.hstack((NNpop - np.sum(soln, axis=1, keepdims=True), soln))

    return soln

param= {    
    'csv_casosBA': "tables/tabelasCasos_Bahia.csv",
    "sizePop": 14930634,
    'rpredict_range': 275,
    'per_incub': 5,  # periodo de incubação em dias 
    'dur_infect_leve': 10, # duração da infeção leve 
    'frac_inf_leve': 0.8, # Fração de infecções em estado leves
    'frac_inf_sev': 0.15, # Fração de infecções em estados severos
    'frac_inf_cri': 0.05, # Fração de infecções em estados criticas ou internados
    'taxa_obit': 0.02,  # Taxa de letalidade (fração de infecções resultando em morte)
    't_intern_obit': 7, # tempo em dias desde que é internado na UTI até a morte
    'dur_hosp': 11, # tempo em dias de internação 
}

 