def edges():
    import pandas as pd
    file_name="auto.xlsx"
    df = pd.read_excel(file_name,sheet_name="Sheet1") # sheet name is needed due to some hidden sheet or something
    df = df.drop('Type',axis=1)
    
    e,i = pd.factorize(df.Source.append(df.Target,ignore_index=True))
    del df
    
    m = e.size//2
    
    import numpy as np
    ee=np.zeros((m,2),dtype=int)
    ee[:,0]=e[:m]
    ee[:,1]=e[m:]
    e=ee
    return e

import igraph as ig
import numpy as np
def undirected_igraph(e=edges(),giant=True):
    # create as undirected, simple graph
    N = np.max(e)+1
    G=ig.Graph(N)
    G.add_edges(list(e))
    G.simplify()

    if giant:
        G=G.components().giant()
    return G

def get_variables(G):
    N = G.vcount()
    m = G.ecount()
    k = np.array(G.degree())
    return (N,m,k)

## degree histogram
#import matplotlib.pyplot as plt
#plt.hist(k,bins=494,log=True)

# layout using gephi

def k_core(G,k=1,fname="k_core.gml"):
    # next step -- k-core
    c=np.array(G.coreness())
    core_ind = list(np.where(c>k)[0])
    G=G.induced_subgraph(core_ind)
    save(G,fname)
    return G, core_ind

def save(G,fname='graph.gml'):
    G.write_gml(fname)

def directed_igraph(e=edges(),giant=False):
    # create as undirected, simple graph
    G=ig.Graph(np.max(e)+1,directed=True)
    G.add_edges(list(e))
    G.simplify(loops=False, combine_edges='sum')

    if giant:
        G=G.components().giant()

    return G
