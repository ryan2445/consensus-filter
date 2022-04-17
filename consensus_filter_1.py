import copy
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 25
NUM_NODES = 10
DIMENSION = 2
MAX_DISTANCE = 4
COMMUNICATION_RADIUS = 2.5
c_v = 0.01
c_2w = (c_v / (COMMUNICATION_RADIUS**2)) / 2
F1 = 50

def isConnected(network):
    #   DFS Helper Function
    def dfs(src, network, visited):
        visited[int(src)] = True

        for nei in network[src]['nei']:
            if visited[int(nei)] == False:
                dfs(nei, network, visited)

    #   Initialize DFS visited
    visited = [False] * NUM_NODES

    #   Begin DFS
    dfs('0', network, visited)

    #   If any node has not been visited, network is not connected
    for visit in visited:
        if not visit:
            return False
    
    #   Network is conected because all nodes visited
    return True

def getNetwork():
    connected = False
    while not connected:
        network = {str(i): {} for i in range(NUM_NODES)}
        nodes = np.random.rand(NUM_NODES, DIMENSION) * MAX_DISTANCE

        #   Initialize network and add neighbors if they are inside communication radius of curent node
        for curr_index, curr_node in enumerate(nodes):
            curr_name = str(curr_index)
            network[curr_name]['pos'] = curr_node
            network[curr_name]['nei'] = []

            for index, node in enumerate(nodes):
                name = str(index)
                if curr_name != name:
                    if np.linalg.norm(curr_node - node) <= COMMUNICATION_RADIUS:
                        network[curr_name]['nei'].append(name)

        connected = isConnected(network)

    #   Plot the network topology
    network_x = [network[node]['pos'][0] for node in network]
    network_y = [network[node]['pos'][1] for node in network]
    plt.scatter(network_x, network_y, c="#1f77b4")
    for name in network:
        node = network[name]
        for nei in node['nei']:
            plt.plot([node['pos'][0], network[nei]['pos'][0]], [node['pos'][1], network[nei]['pos'][1]], c="#1f77b4")
    plt.savefig('network.png')
    plt.clf()

    #   Calculate average position of all nodes on network
    avgPos = np.sum([network[name]['pos'] for name in network], axis=0) / NUM_NODES
    
    #   Return the randomly generated connected network of nodes
    return network, avgPos

network, avgPos = getNetwork()
x_i = F1 * np.ones((NUM_NODES, 1)) + 1 * np.random.randn(NUM_NODES, 1)
x_i_initial = copy.copy(x_i)
x_i_old = copy.copy(x_i)
convergence = {i: F1 * np.ones((ITERATIONS, 1)) for i in range(NUM_NODES)}
weights = {str(i): {str(j): 0 for j in network} for i in network}

def c_1w(len_nei):
    return (2 * c_v) / (COMMUNICATION_RADIUS**2 * len_nei)

def v_i(pos):
    return ((np.linalg.norm(pos - avgPos)**2) + c_v) / (COMMUNICATION_RADIUS**2)

for i in range(1, ITERATIONS):
    for name in network:
        sigma_weight = 0
        for nei in network[name]['nei']:
            # WEIGHT DESIGN 1
            # weights[name][nei] = c_1w(len(network[name]['nei'])) / (v_i(network[name]['pos']) + v_i(network[nei]['pos']))
            # WEIGHT DESIGN 2
            # weights[name][nei] = (1 - weights[name][name]) / len(network[name]['nei'])
            # MAX-DEGREE WEIGHTS
            
            sigma_weight += weights[name][nei]

        # WEIGHT DESIGN 1
        # weights[name][name] = 1 - sigma_weight
        # WEIGHT DESIGN 2
        # weights[name][name] = c_2w / v_i(network[name]['pos'])
        # MAX-DEGREE WEIGHTS


        addition = 0
        for nei in network[name]['nei']:
            addition += weights[name][nei] * x_i_old[int(nei)]
        
        x_i[int(name)] = weights[name][name] * x_i_old[int(name)] + addition
    
    for j in range(len(x_i)):
        convergence[j][i] -= x_i[j]

    x_i_old = copy.copy(x_i)

#   Plot measurements
measurement_x_initial = [i+1 for i in range(NUM_NODES)]
measurement_y_initial = x_i_initial
measurement_x_final = [i+1 for i in range(NUM_NODES)]
measurement_y_final = x_i
plt.plot(measurement_x_initial, measurement_y_initial, label="Initial Measurement")
plt.plot(measurement_x_final, measurement_y_final, label="Final Measurement")
plt.xlabel("Node")
plt.ylabel("Measurement Value")
plt.savefig("measurements.png")
plt.clf()

convergence_x = [i for i in range(1, ITERATIONS)]
convergence_x
for i in range(NUM_NODES):
    plt.plot(convergence_x, np.delete(convergence[i].flatten(), 0))
plt.savefig("convergence.png")
plt.clf()
