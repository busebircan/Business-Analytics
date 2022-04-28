# A metaheuristics approach for different sizes of single allocation p-hub location problem
Uncapacitated single allocation p-hub location problem on CAB, TR and randomly generated datasets which are larger is covered. The objective is to minimise total transportation cost of a network while satisfying allocated demand flow by solving this NP-hard problem using metaheuristics. Decision-making for the location of hubs and allocation of non-hubs to the selected hubs is focused on. A simulated annealing and a genetic algorithm are applied to the networks with different sizes and with various pre-determined numbers of hubs. Finally, computational results are reported comparing methods performances, numbers of hubs and suggested location-allocation solutions in terms of objective function.
Simulated annealing and genetic algorithms are implemented as approximation techniques, initializing with random feasible solutions.
# Neighbourhood Structures
Four different neighbourhood structures are built and used in both algorithms to move from one solution to another:
1. changing a hub with a node allocated to it,
2. changing two nodes’ hubs with each other,
3. changing nodes’ hub to another keeping the hubs same,
4. changing allocated node groups between hubs keeping the hubs same.
# Conclusions
SA algorithm was found efficient with an average of 7.3% of gap from known best solutions on average of all networks. For most of the network sizes, the SA obtained good solutions with significantly small gaps to known optimal solutions in only 10 test runs. Moreover, good solutions were achieved in shorter computational times despite the need to increase number of iterations as network becomes larger.

GA was run with smaller number of generations relative to number of iterations in SA in order to decrease runtimes for the problem. This caused GA to perform worse than desired especially for larger networks and it could not improve the solutions recommended by SA algorithm. However, obtained results’ gaps with the known optimal solutions under less than 100 generations even for the largest network tested, gives good insights for the availability of obtaining much better results when more computational time can be afforded.
