import matplotlib.pyplot as plt
import numpy as np

NUM_NODES = 10
DIMENSION = 2
MAX_DISTANCE = 4
COMMUNICATION_RADIUS = 1.6
NOISE = 0.01

def isConnected(network):
    def dfs(src, network, visited):
        visited[int(src)] = True

        for nei in network[src]['nei']:
            if visited[int(nei)] == False:
                dfs(nei, network, visited)

    visited = [False] * NUM_NODES

    dfs('0', network, visited)

    for visit in visited:
        if not visit:
            return False
    
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

def sensor_reading(network, name, avgPos):
    return (np.linalg.norm(network[name]['pos'] - avgPos) ** 2 + NOISE) / (COMMUNICATION_RADIUS ** 2)

network, avgPos = getNetwork()

# readings = {str(i): 0 for i in range(NUM_NODES)}
# design_factor = 0.5
# for i in range(1, 100):
#     for name in network:

#         for nei in network[name]['nei']:


#variance = (np.linalg.norm(network['pos'] - avgPos) ** 2 + NOISE) / (COMMUNICATION_RADIUS ** 2)

