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