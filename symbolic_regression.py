# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:36:27 2020

@author: Karlo
"""
from gplearn.genetic import SymbolicRegressor
import pandas as pan
import numpy as np
import random
from sklearn.model_selection import train_test_split

#Importiranje podataka i podjela na ulazne i izlazne parametre  
data = pan.read_csv("parkinsons_updrs.data").to_numpy()
X = data[:,(1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)]
Y_total = data[:,5]
Y_motor = data[:,4]

X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X, Y_total,random_state=1)
X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X, Y_motor,random_state=1)

def coefficient_genetic_operations(crosover,mutation):
    p_crossover=0
    p_subtree=0
    p_hoist=0
    p_point=0
    while(p_crossover + p_subtree + p_hoist + p_point != 1):
        p_crossover=random.choice(crosover)
        p_subtree = random.choice(mutation)
        p_hoist = random.choice(mutation)
        p_point = random.choice(mutation)
        
    return p_crossover, p_subtree, p_hoist, p_point
        
#Inicjalizacija populacije(nasumicnim odabirom popunjavanje parametara neuronske mreže)
def initialization_population(size_mlp):
    population_size = [1000,2000,3000]
    generations = [50,100,150,200,250]
    tournament_size = [50,100,200,500]
    stopping_criteria=[0,0.1,0.01,0.001]
    const_range = [(-1,1),(-100,100),(-1000,1000)]
    init_depth = [(3,6),(7,12)]
    function_set=['add','sub','mul','div','sqrt','log','abs','max','min','sin','cos','tan']
    parsimony_coefficient=[0.0001,0.001,0.01]   
    p_crossover=[0.7,0.8,0.9,1]
    p_mutation=[0,0.1,0.2,0.3]
    max_samples=[0.9,1]
    
    coefficients=coefficient_genetic_operations(p_crossover,p_mutation)

    pop =  np.array([[random.choice(population_size), random.choice(generations),
                      random.choice(tournament_size), random.choice(stopping_criteria),
                      random.choice(const_range),random.choice(init_depth),
                      function_set,random.choice(parsimony_coefficient),
                      coefficients[0],coefficients[1],coefficients[2],coefficients[3],
                      random.choice(max_samples)]])
    
    for i in range(0, size_mlp-1):
        coefficients=coefficient_genetic_operations(p_crossover,p_mutation)
        pop = np.append(pop,[[random.choice(population_size), random.choice(generations),
                      random.choice(tournament_size), random.choice(stopping_criteria),
                      random.choice(const_range),random.choice(init_depth),
                      function_set,random.choice(parsimony_coefficient),
                      coefficients[0],coefficients[1],coefficients[2],coefficients[3],
                      random.choice(max_samples)]] , axis=0)
    return pop

def crossover(par_1, par_2):
    child = [par_1[0], par_2[1], par_1[2], par_2[3],par_1[4],par_2[5],par_1[6],par_2[7],par_1[8],par_1[9],par_1[10],par_1[11],par_2[12]]    
    return child

#Mutacija broja neurona,Sto je veći prob_mut to su šanse manje za mutaciju
def mutation(child, prob_mut):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() > prob_mut:
            child_[c,1] = int(child_[c,1]) + random.randint(10, 20)
    return child_

#Računanje fitnes funkcije na način da se izračuna točnost mreže za zadane parametre
def fitness_function(pop, X, Y): 
    fitness = []
    for w in pop:
        est_gp = SymbolicRegressor(population_size=(int(w[0])),
                           generations=(int(w[1])),
                           tournament_size=(int(w[2])), 
                           stopping_criteria=(float(w[3])),
                           const_range = w[4],
                           init_depth =w[5],
                           function_set=w[6],
                           parsimony_coefficient=(float(w[7])),
                           p_crossover=(float(w[8])),
                           p_subtree_mutation=(float(w[9])),
                           p_hoist_mutation=(float(w[10])),
                           p_point_mutation=(float(w[11])),
                           max_samples=(float(w[12])),
                           verbose= 1)    
        try:
            est_gp.fit(X, Y)
            formula=est_gp._program
            score=est_gp._program.raw_fitness_
            print("Score:"+ str(score))
            fitness.append([score, w, formula])
        except:
            pass
        
    return fitness

def genetic_algorithm(X, Y, num_epochs, size_mlp, prob_mut):
    pop = initialization_population(size_mlp)
    fitness = fitness_function(pop, X, Y)
    pop_fitness_sort = np.array(list(sorted(fitness,key=lambda x: x[0])))
    
    for j in range(0, num_epochs):
        length = len(pop_fitness_sort)
        
        #Izabir roditelja
        parent_1 = pop_fitness_sort[:,1][:length//2]
        parent_2 = pop_fitness_sort[:,1][length//2:]
    
        #Krizanje i mutacija
        child_1 = [crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [crossover(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutation(child_2, prob_mut)

        #Izračun fitnesa
        fitness_child_1 = fitness_function(child_1, X, Y)
        fitness_child_2 = fitness_function(child_2, X, Y)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(sorted(pop_fitness_sort,key=lambda x: x[0])))
        #Izabir individualaca za sljedeću generaciju
        pop_fitness_sort = sort[0:size_mlp, :]
        
    return sort

results_total  = genetic_algorithm(X_train_total, y_train_total, num_epochs = 10, size_mlp = 20, prob_mut = 0.9)

with open('Results_Total.txt', 'w') as f:
    for i in range(len(results_total)):
        print(str(results_total[i][0])+"\n"+str(results_total[i][1])+"\n"+str(results_total[i][2]), file=f)
        print("-------------------------------------------------------", file=f)
        
with open('Score_and_Formulas_Total.txt', 'w') as f1:
    for i in range(len(results_total)):
        print(str(results_total[i][0])+","+str(results_total[i][2]), file=f1)
        
with open('Formulas_Total.txt', 'w') as f2:
    for i in range(len(results_total)):
        print(str(results_total[i][2]),file=f2)
        
results_motor  = genetic_algorithm(X_train_motor, y_train_motor, num_epochs = 10, size_mlp = 20, prob_mut = 0.9)


with open('Results_Motor.txt', 'w') as f:
    for i in range(len(results_motor)):
        print(str(results_motor[i][0])+"\n"+str(results_motor[i][1])+"\n"+str(results_motor[i][2]), file=f)
        print("-------------------------------------------------------", file=f)
        
with open('Score_and_Formulas_Motor.txt', 'w') as f1:
    for i in range(len(results_motor)):
        print(str(results_motor[i][0])+","+str(results_motor[i][2]), file=f1)
        
with open('Formulas_Motor.txt', 'w') as f2:
    for i in range(len(results_motor)):
        print(str(results_motor[i][2]),file=f2)