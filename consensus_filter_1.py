import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 10
NUM_NODES = 10
DIMENSION = 2
MAX_DISTANCE = 4
COMMUNICATION_RADIUS = 1.6
c_v = 0.01
c_lw = (2 * c_v) / (COMMUNICATION_RADIUS**2 * (NUM_NODES - 1))
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
                    if np.linalg.norm(curr_node - node) <= 1.6:
                        network[curr_name]['nei'].append(name)

        connected = isConnected(network)

    #   Plot the network topology
    network_x = [network[node]['pos'][0] for node in network]
    network_y = [network[node]['pos'][1] for node in network]
    plt.scatter(network_x, network_y)
    for name in network:
        node = network[name]
        for nei in node['nei']:
            plt.plot([node['pos'][0], network[nei]['pos'][0]], [node['pos'][1], network[nei]['pos'][1]])
    plt.savefig('test.png')

    #   Calculate average position of all nodes on network
    avgPos = np.sum([network[name]['pos'] for name in network], axis=0) / NUM_NODES
    
    #   Return the randomly generated connected network of nodes
    return network, avgPos

network, avgPos = getNetwork()
nodes_va = 50 * np.ones((NUM_NODES,1)) + 1 * np.random.randn(NUM_NODES, 1)
nodes_va_old = nodes_va
n1 = [0] * NUM_NODES
m1 = [0] * NUM_NODES
x

weights = {str(i): {str(j): 0 for j in network} for i in network}

def v_i(pos, node):
    v1 = ((np.linalg.norm(pos - avgPos)**2) + c_v) / (COMMUNICATION_RADIUS**2)
    
    n1[node] = np.random.normal(0.0, v1)

    m1[node] = F1 + n1[node]

    return v1

for i in range(1, ITERATIONS):
    for name in network:
        sigma_weight = 0
        for nei in network[name]['nei']:
            weights[name][nei] = c_lw / (v_i(network[name]['pos'], int(name)) + v_i(network[nei]['pos'], int(name)))
            sigma_weight += weights[name][nei]
        weights[name, name] = 1 - sigma_weight

        addition = 0
        for nei in network[name]['nei']:
            addition += weights[name][nei] * nodes_va_old[int(nei)]
        
        nodes_va[int(name)] = weights[name][name] * nodes_va_old[int(name)] + addition
    
    nodes_va_old = nodes_va


