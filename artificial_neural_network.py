import numpy as np
import pandas as pan
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Importiranje podataka i podjela na ulazne i izlazne parametre
data = pan.read_csv("parkinsons_updrs.data").to_numpy()
X = data[:,(1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)]
Y_total = data[:,5]
Y_motor = data[:,4]


X_train_total1, X_test_total1, y_train_total, y_test_total = train_test_split(X, Y_total,random_state=1)
X_train_motor1, X_test_motor1, y_train_motor, y_test_motor = train_test_split(X, Y_motor,random_state=1)
scaler = StandardScaler()
scaler.fit(X_train_total1)
X_train_total = scaler.transform(X_train_total1)
X_test_total = scaler.transform(X_test_total1)
scaler2 = StandardScaler()
scaler2.fit(X_train_motor1)
X_train_motor = scaler.transform(X_train_motor1)
X_test_motor = scaler.transform(X_test_motor1)


#Inicjalizacija populacije(nasumicnim odabirom popunjavanje parametara neuronske mreže)
def initialization_population(size_mlp):
    activation = ['logistic', 'tanh', 'relu']
    solver = ['lbfgs','sgd', 'adam']
    numberOfNeurons=[50,75,100]
    alpha = [0.1,0.00001, 0.0001, 0.001,0.01,0]
    pop =  np.array([[random.choice(activation), random.choice(solver),random.choice(alpha), random.choice(numberOfNeurons),random.choice(numberOfNeurons),random.choice(numberOfNeurons),random.choice(numberOfNeurons)]])
    for i in range(0, size_mlp-1):
        pop = np.append(pop, [[random.choice(activation), random.choice(solver),random.choice(alpha),random.choice(numberOfNeurons),random.choice(numberOfNeurons),random.choice(numberOfNeurons),random.choice(numberOfNeurons)]], axis=0)
    return pop

#Jedinstveno križanje kromosoma
def crossover(par_1, par_2):
    child = [par_1[0], par_2[1], par_1[2], par_2[3],par_1[4],par_2[5],par_2[6]]    
    return child

#Mutacija broja neurona,Sto je veći prob_mut to su šanse manje za mutaciju
def mutation(child, prob_mut):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() > prob_mut:
            k = random.randint(3,6)
            child_[c,k] = int(child_[c,k]) + random.randint(5, 10)
    return child_

#Računanje fitnes funkcije na način da se izračuna točnost mreže za zadane parametre
def fitness_function(pop, X_train, y_train, X_test, y_test): 
    fitness = []
    for w in pop:
        clf = MLPRegressor(alpha=float(w[2]),hidden_layer_sizes=(int(w[3]),int(w[4]), int(w[5]), int(w[6])), 
                    max_iter = 300000 ,max_fun=300000, activation = w[0], verbose = False,  n_iter_no_change=700,
                    solver = w[1], learning_rate = 'adaptive')
        clf.fit(X_train,y_train)
        try:
            score=clf.score(X_test, y_test)
            print("Score:"+ str(score))
            fitness.append([score, clf, w])
        except:
            pass
        
    return fitness

def genetic_algorithm(X_train, y_train, X_test, y_test, num_epochs, size_mlp, prob_mut):
    pop = initialization_population(size_mlp)
    fitness = fitness_function(pop,  X_train, y_train, X_test, y_test)
    pop_fitness_sort = np.array(list(reversed(sorted(fitness,key=lambda x: x[0]))))

    for j in range(0, num_epochs):
        length = len(pop_fitness_sort)
        
        #Izabir roditelja
        parent_1 = pop_fitness_sort[:,2][:length//2]
        parent_2 = pop_fitness_sort[:,2][length//2:]
        
        #Krizanje i mutacija
        child_1 = [crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [crossover(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutation(child_2, prob_mut)
    
        #Izračun fitnesa
        fitness_child_1 = fitness_function(child_1,X_train, y_train, X_test, y_test)
        fitness_child_2 = fitness_function(child_2, X_train, y_train, X_test, y_test)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(reversed(sorted(pop_fitness_sort,key=lambda x: x[0]))))
        
        #Izabir individualaca za sljedeću generaciju
        pop_fitness_sort = sort[0:size_mlp, :]
        best_individual = sort[0][1]
        
    return best_individual

Best_architecture_total = genetic_algorithm(X_train_total, y_train_total, X_test_total, y_test_total, num_epochs = 10, size_mlp = 20, prob_mut = 0.9).fit(X_train_total,y_train_total)
Final_score_total =Best_architecture_total.score(X_test_total, y_test_total)

Best_architecture_motor = genetic_algorithm(X_train_motor, y_train_motor, X_test_motor, y_test_motor, num_epochs = 10, size_mlp = 20, prob_mut = 0.9).fit(X_train_motor,y_train_motor)
Final_score_motor =Best_architecture_motor.score(X_test_motor, y_test_motor)

print("Final Score total:"+ str(Final_score_total))
print("Final Score motor:"+ str(Final_score_motor))
