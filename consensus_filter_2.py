import copy
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 25
NUM_NODES = 30
DIMENSION = 2
MAX_DISTANCE = 25
COMMUNICATION_RADIUS = 5
c_v = 0.01
c_2w = (c_v / (COMMUNICATION_RADIUS**2)) / 2

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
    
    return network

def parseField():
    file = open('Scalar_Field_data2.txt')
    lines = file.readlines()
    file.close()

    parsed = [0] * len(lines)

    for i in range(len(lines)):
        parsed[i] = list(map(float, lines[i].replace("\n", "").split("\t")))

    return np.array(parsed)

network = getNetwork()
F1 = parseField()
x_i = F1
x_i_initial = copy.copy(x_i)
x_i_old = copy.copy(x_i)
weights = {str(i): {str(j): 0 for j in network} for i in network}

def v_i(pos, field_location):
    return ((np.linalg.norm(pos - field_location)**2) + c_v) / (COMMUNICATION_RADIUS**2)

for x in range(25):
    for y in range(25):
        for i in range(1, ITERATIONS):
            for name in network:
                for nei in network[name]['nei']:
                    # WEIGHT DESIGN 2
                    weights[name][nei] = (1 - weights[name][name]) / len(network[name]['nei'])

                # WEIGHT DESIGN 2
                weights[name][name] = c_2w / v_i(network[name]['pos'], [x, y])

                addition = 0
                for nei in network[name]['nei']:
                    addition += weights[name][nei] * x_i_old[x][y]
                
                x_i[x][y] = weights[name][name] * x_i_old[x][y] + addition

            x_i_old = copy.copy(x_i)

#   Plot the measured scalar field
plt.imshow(x_i, cmap='hot', interpolation='nearest')
plt.savefig("heatmap.png")
plt.clf()

diff = x_i - x_i_initial
diff_x = [i+1 for i in range(625)]
diff_y = diff.flatten()
plt.plot(diff_x, diff_y)
plt.savefig("heatmap_error.png")
plt.clf()
