# script to convert smiles to graph
import sys
try:
    from pysmiles import read_smiles

    import networkx as nx

    import time

    import numpy as np

    import pickle

    import re

    start_time = time.time()        #Start time of execution


    GRAPH = []                      #to store graph data
    i = 0                           #for counting reactions
    _GRAPH = []                     # to store reshaped graph
    rr = []

    def smiles_to_graph(sml):

        rx= sml                         #for storing smiles of sing reaction
        global GRAPH                             #for assigning labels
        global a
        global c
        global rr

        mol = read_smiles(rx)
        gph = nx.adjacency_matrix(mol, weight='order').todense()
        graph = np.matrix(gph)
        GRAPH.append(graph)

    # reshape graph
    ########################################
    def reshape_graph(GRAPH):
        sh = []
        global _GRAPH
        m = 0
        for g in GRAPH:
            if g.shape[0] == g.shape[1]:
                sh.append(g.shape[0])
        m = 86
        for G in GRAPH:
            if G.shape[0] <= m:
                g_sh = G.shape[0]
                print(g_sh)
                r = m - g_sh
                for i in range(r):
                    G = np.hstack((G, np.atleast_2d(np.zeros(g_sh)).T))
                for i in range(r):
                    G = np.vstack((G, np.zeros(m)))
                _GRAPH.append(G)

        for G in _GRAPH:
            print(G.shape)

    # start to read file and calling convert function
    i_file_name = input('Enter input file name: ')
    i_file_name = str(i_file_name)
    l_file_name = input('Enter output file name (with .data extension) for saving lables: ')
    l_file_name = str(l_file_name)
    g_file_name = input('Enter output file name (with .data extension) for saving graph: ')
    g_file_name = str(g_file_name)

    sm = open(i_file_name, "r")
    #smiles = iter(sm)               # to skip the first entry
    #next(smiles)
    labels = []
    ll = 0
    for s in sm:
        label = int(re.search(r'\d+', s[:7]).group())                   # to extract label
        labels.append([label])
        ll = ll + 1
        print(ll)
        smiles_to_graph(s[6:].replace(" ", ""))                         # to call conversion function
    sm.close()

    reshape_graph(GRAPH)                                        #=> calling reshape function

    #print(labels)                                                   # to save labels

    with open(l_file_name, 'ab') as filehandle:
        pickle.dump(labels, filehandle)
    filehandle.close()


    with open(g_file_name, 'ab') as filehandle:
        #store the data as binary data stream
        pickle.dump(_GRAPH, filehandle)
    filehandle.close()

    print(np.array(_GRAPH).shape)
    print(np.array(GRAPH).shape)

    print("--- %s seconds ---" % (time.time() - start_time))
except Exception as e:
    print(e)
finally:
    print("Terminating this code..!") 
