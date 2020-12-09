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

# @Ii: número de infectados (leve, severo, graves)
# @t : tempo
# @beta: taxa de infecção
# @taxa_prog: taxa de propagação [a]
# @gamma: taxa de recuperação
# @taxa_infec: 

def modeloSEIR(Ii, t, beta, taxa_prog, gamma, taxa_infec, taxa_obito, N): 
    
    # Individuos suceptiveis    
    Sscept = N - sum(Ii)     ### N = Sscept + E + I1 + I2 + I3 + R + D
    # individuos expostos
    indExp = Ii[0]

    ##################################################
    ##### Equações diferenciais do modelo ############    
    
    df_Sscept = -1 * ((beta[1] * Ii[1]) - (beta[2] * Ii[2]) - (beta[3] * Ii[3])) * Sscept
    # individuos expostos [infectados, mas ainda não infecciosos ou sintomáticos] 
    df_Exp= np.dot(beta[1:3], Ii[1:3]) * Sscept - (taxa_prog * indExp)    # espostos 

    ### Infecção leve (não é necessária hospitalização)
    df_infL= (taxa_prog * indExp) - (gamma[1] + taxa_infec[1]) * Ii[1] #I1
    
    ### Infecção grave (hospitalização necessária)
    df_infG = (taxa_infec[1] * Ii[1]) - (gamma[2] + taxa_infec[2]) * Ii[2] #I2
    
    ### Infecção crítica (UTI necessária)
    df_infC = (taxa_infec[2] * Ii[2]) - (gamma[3] + taxa_obito) * Ii[3] #I3

    ### indivíduos que se recuperaram da doença 
    df_recup= np.dot(gamma[1:3], Ii[1:3]) #R
    ### Dead individuals
    df_obit= taxa_obito * Ii[3] #D

    return [df_Sscept, df_Exp, df_infL, df_infG, df_infC, df_recup, df_obit]

##########################################################
#### calculando parametros de taxa de reproduçã R0 #######
##########################################################
def calculo_taxa_repR0(pmts):
    
    Npop = pmts['sizePop'] 
    beta = np.zeros(4)
    gamma = np.zeros(4)
    
    # taxas de infeção em individuos hospitalizados
    taxaInf = np.zeros(3)  #[p]

    ## periodo de encubação 
    tax_prog = 1 / pmts['per_incub']    # alpha
    
    valorTemp = 1 / pmts['t_intern_obit']
    tax_Ob = valorTemp * (pmts['taxa_CFR'] / pmts['frac_inf_cri'])   # u  miu 
    gamma[3] = valorTemp - tax_Ob

    valorTemp = 1 / pmts['dur_hosp']    
    taxaInf[2] =  valorTemp * (pmts['frac_inf_cri'] / (pmts['frac_inf_cri'] +  pmts['frac_inf_sev']))    
    gamma[2] = valorTemp - taxaInf[2]

    # print(gamma[2])
    # taxa de infeção dos obitos 
    valorTemp = 1 / pmts['dur_infect_leve']    
    gamma[1] = valorTemp * pmts['frac_inf_leve']    
    taxaInf[1] =  valorTemp - gamma[1]    # p
    # print("valor temporal 2 ", valorTemp)
    # print(gamma[1])
    # print(taxaInf[1])
    
    #beta= 2e-4 * np.ones(4) # todos os estágios transmitem igualmente
    beta = 2.5e-4 * np.array([0,1,1,1]) # casos hospitalizados não transmitem   

    # Calculo do radio de infeção basico 
    prop1 = Npop / (taxaInf[1] + gamma[1])     
    print("primeiro termo ", prop1)
    
    prop2 = taxaInf[1]  / (taxaInf[2] + gamma[2])
    print('segundo termo ', prop2)
    
    prop3 = taxaInf[2] /  (tax_Ob + gamma[3])
    print("terceiro termo ", prop3)
    
    R0 = prop1 * (beta[1] + prop2 * (beta[2] + beta[3] * prop3))            
    
    return R0, Npop, tax_prog, beta, gamma, taxaInf, tax_Ob


def trainingModelSIER (pmts, dfger):
    
    tmax = dfger.shape[0]
    # tempo em vetores 
    tvec= np.arange(0,tmax)
    # print("#############################################")
    print(tmax)
    print("tamanho do vector de tempo [tvec]: ", len(tvec))
        
    R0, NNpop, atax_prog, bbeta, ggamma, ptaxaInf, utax_Ob = calculo_taxa_repR0(pmts)
    # N = Sscept + E + I1 + I2 + I3 + R + D
    ic = np.zeros(6)
    ic[0]= 1    



    print("O radio de infeão basico : {0:4.3f}". format(R0))
    print("população da Bahia: [N] = {}".format(NNpop))
    print("taxa de progresão da infecção [alpha] = {}".format(atax_prog))
    print("taxa de infecção a susceptiveis [beta]: \n b1_{}, b2_{},  b3_{}".format(bbeta[0], bbeta[1], bbeta[2]))
    print("taxa de infeção em individuos em recuperação [gamma]: \n g1_{}, g2_{},  g3_{}".format(
                        ggamma[0], ggamma[1], ggamma[2]))
    print("taxas de infeção em individuos hospitalizados [p]: \n p1_{}, p2_{},  p3_{}".format(
                        ptaxaInf[0], ptaxaInf[1], ptaxaInf[2]))
    print("taxa de obitos [u] = {}".format(utax_Ob))

    #                          Ii, t,          beta, taxa_prog, gamma, taxa_infec, taxa_obito, N
    soln = odeint(modeloSEIR, ic, tvec, args=(bbeta, atax_prog, ggamma, ptaxaInf, utax_Ob, NNpop))
    
    print("modelo Odeint treinado ", soln)
    print("tamanho ", len(soln))
    soln = np.hstack((NNpop - np.sum(soln, axis=1, keepdims=True), soln))

    plt.figure(figsize=(30, 7))
    plt.subplot(1,2,1)
    plt.plot(tvec, soln)
    plt.xlabel("Time (days)")
    plt.ylabel("Number per 1000 People")
    plt.legend(("Sus","Exp","I1","I2","I3","Rec","D_obit"))
    plt.ylim([0, NNpop + 1000])
    plt.show()

    return soln

param= {    
    'csv_casosBA': "tables/tabelasCasos_Bahia.csv",
    "sizePop": 14873064,    
    'per_incub': 5,  # periodo de incubação em dias 
    'dur_infect_leve': 10, # duração em dias da infeção leve 
    'frac_inf_leve': 0.8, # Fração de infecções em estado leves
    'frac_inf_sev': 0.15, # Fração de infecções em estados severos
    'frac_inf_cri': 0.05, # Fração de infecções em estados criticas ou internados
    'taxa_CFR': 0.02,  # Taxa de letalidade (fração de infecções resultando em morte)
    't_intern_obit': 7, # tempo em dias desde que é internado na UTI até a morte
    'dur_hosp': 11, # tempo em dias de internação 
}

 

print("#####################################################################")
print("###########      inicializando o processo de training     ###########")
print("#####################################################################")

df_covid = pd.read_csv(param['csv_casosBA'])
print(df_covid.columns)
print(df_covid.tail(10))

trainingModelSIER(param, df_covid)