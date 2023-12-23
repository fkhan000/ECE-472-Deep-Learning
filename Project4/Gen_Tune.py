import numpy as np

#########REAL_PARAMS CLASS#########

#This class holds all of the real valued parameters in our model
#like the step size and regularization parameter.

class Real_Params():

    def __init__(self, val):
        
        self.val = val

    #This function performs crossover between two real
    #valued parameters and produces two children from them
        
    def crossover(self, parent2):
        rng = np.random.default_rng(0x3942e)

        #we randomly select a weighting coefficient
        beta = rng.uniform(-0.25, 1.25, len(self.val))

        #and then we create two Real_Params objects from the parent
        #objects whose values are the weighted averages of the values
        #of their parents
        child1 = Real_Params(self.val*beta + parent2.val*(1-beta))
        child2 = Real_Params(parent2.val*beta + self.val*(1-beta))

        return child1, child2

#This class holds the discrete parameters of our model that
#have the same minimum and maximum values. 
class Discrete_Params():

    def __init__(self, val, maxi):

        self.val = val
        self.maxi = maxi

    #This function performs crossover by randomly selecting
    #parameters from either parent
    def crossover(self, parent2):

        rng = np.random.default_rng(0x3942e)

        #We create an array of 0s and 1s
        beta = rng.uniform(0, 1, len(self.val))

        beta[beta <= 0.5] = 0

        beta[beta > 0.5] = 1

        mutation = rng.uniform(0, 1, len(self.val))

        mutation[mutation <= 0.5] = 0
        mutation[mutation > 0.5] = 1

        #and then we do sort of like a weighted sum so that we get
        #each of the parameters either from parent1 or parent2
        child1 = Discrete_Params(np.mod(self.val*beta + parent2.val*(1-beta) + mutation, self.maxi), self.mini, self.maxi)
        child2 = Discrete_Params(np.mod(parent2.val*beta + self.val*(1-beta) + mutation, self.maxi), self.maxi)

                       

        return child1, child2


#This function creates the next generation of parameters
#It takes in the current generation, the age limit (when
#individuals in the population are removed, and the fitness
#function)
def generate(population, age, fit_func):

    #we will perform num_add crossover operations for this generation
    num_add = len(population) // 8
    rng = np.random.default_rng(0x3942e)

    
    for count in range(num_add):

        #we randomly sample 5 individuals from the population twice
        selection  = rng.choice(population, size = [2, 5])

        #and then get the fittest individual from each selection
        parentA = min(selection[0], key = lambda x: x[1])[0]
        parentB = min(selection[1], key = lambda x: x[1])[0]
        
        child1 = np.asarray([])
        child2 = np.asarray([])

        #for each of the parents' parameters
        for param1, param2 in zip(parentA, parentB):

            #we perform crossover
            children = param1.crossover(param2)

            #and add it to the children produced by the crossover
            child1 = np.append(child1, children[0])
            child2 = np.append(child2, children[1])

        #then we add child1 and child2 to the population
        population += [[child1, fit_func(child1), 0],  [child2, fit_func(child2), 0]]

    #we sort the population by their ages in ascending order
    population.sort(key = lambda x: x[2])
    index = 0

    #then we increment the ages of everyone in the population
    while(index < len(population) and population[index][2] != age):

        population[index][2] += 1
        index += 1

    #and then remove all of the indiivduals that exceed the age limit
    #This keeps the population size stable and for an age limit of 3
    #and 1.25*population individuals born every generation, this should
    #cause the population size to oscillate and be bounded
    return population[:index]

        
    

#This function is the main driver for our tuning algorithm
#Init_size is the initial size of our population, num_gen is the number
#of generations we should run for, fit_func our fitness function, and
#parameters encodes the parameters that we need to tune for
def tuning(init_size, num_gen, fit_func,  parameters, age = 3):
    
    rng = np.random.default_rng(0x3942e)
    population = []

    #for each individual
    for pop in range(init_size):

        individual = np.asarray([])

        #we go through our list of parameters
        for param in parameters:

            #if the first element is 0
            if param[0] == 0:
                #the parameter is discrete and we choose its value randomly from the range provided in the parameter list
                individual = np.append(individual, Discrete_Params(rng.choice(list(range(int(param[1]), int(param[2]) + 1)), size = int(param[3])), int(param[2] + 1)))

            elif param[0] == 1:
                #else if the first element is 1, then we have a real valued parameter. So we get the value from a
                #uniform distribution between the min and max value given
                individual = np.append(individual, Real_Params(rng.uniform(param[1], param[2], int(param[3]))))

        #we then add the individual to our population
        population.append([individual, fit_func(individual),  1])


    #for each generation
    for iteration in range(num_gen):

        #we get the highest performing set of parameters
        best = min(population, key = lambda x: x[1])

        #and display its fitness (which in this case is the error it gets on the validation set
        #after training)
        print("\n\nGeneration Count: " + str(iteration) + "\n\n")
        print("Best Score: " + str(best[1]) + "\n\n")

        #then we generate the next generation
        population =generate(population, age, fit_func)


    best = min(population, key = lambda x: x[1])

    #and we then return the most fit individual
    return best[0]
