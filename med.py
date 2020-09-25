import pandas as pd
import igraph as ig
import numpy as np

def get_df(cached=True,save=True):
    if cached:
        df = pd.read_hdf('kayvan_data.h5')
    else:
        file_name='dat/Backward SC 2020-09-19 Tiers 1,2,3,4,5 Supplier Only In one Column (1).xlsx'
        df = pd.read_excel(file_name,sheet_name="Tier 1-5")
        df = df.drop('Type',axis=1)
        df = df.drop_duplicates(ignore_index=True)
        if save:
            df.to_hdf('kayvan_data.h5',key='df')
    return df

def get_attributes():
    df = get_df()

    firm_df = pd.DataFrame(dict(ID=df['Source'],name=df['Source NAME'],industry=df['Source Industry'],country=df['Source Country']))
    firm_df = firm_df.append(pd.DataFrame(dict(ID=df['Target'],name=df['Target NAME'],industry=df['Target Industry'],country=df['Target Country'])))
    firm_df = firm_df.drop_duplicates()
    firm_df = firm_df.set_index('ID')

    return firm_df

def edges():
    df=get_df()
    e,_ = pd.factorize(df.Source.append(df.Target,ignore_index=True))
    
    m = e.size//2
    
    ee=np.zeros((m,2),dtype=int)
    ee[:,0]=e[:m]
    ee[:,1]=e[m:]
    e=ee
    return e

def undirected_igraph(e=None,giant=True):
    if e is None: e=edges() # can't have this as a default argument, since these are evaluated at import time, and it is time consuming
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

def k_core(G,k=1,fname="k_core.gml"):
    # next step -- k-core
    c=np.array(G.coreness())
    core_ind = list(np.where(c>k)[0])
    G=G.induced_subgraph(core_ind)
    save(G,fname)
    return G, core_ind

def save(G,fname='graph.gml'):
    G.write_gml(fname)

def directed_igraph(e=None,giant=False):
    # create as undirected, simple graph
    if e is None: e=edges() # can't have this as a default argument, since these are evaluated at import time, and it is time consuming
    G=ig.Graph(np.max(e)+1,directed=True)
    G.add_edges(list(e))
    G.simplify(loops=False, combine_edges='sum')
    firm_df=get_attributes()
    G.vs['name']=x[0]
    G.vs['industry']=x[1]
    G.vs['country']=x[2]
    G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs

    if giant:
        G=G.components().giant()

    return G

def get_terminal_suppliers(i,G):
    import itertools

    d = G.bfs(i,mode='IN')
    d = d[0][:d[1][-1]] # remove trailing zeros
    d = G.induced_subgraph(d)

    sccs = d.clusters()
    terminal_components = sccs.cluster_graph().vs(_indegree_eq = 0)
    terminal_components = [i.index for i in terminal_components]

    sccs_l = list(sccs)
    terminal_nodes = [sccs_l[i] for i in terminal_components]
    terminal_nodes = itertools.chain(*terminal_nodes)
    terminal_nodes = [d.vs[i]['id'] for i in terminal_nodes]

    return terminal_nodes

# i is an integer, the index of the node in G
def some_terminal_suppliers_reachable(i,G,G_thin):
    t = get_terminal_suppliers(i,G)

    try:
        i_thin = G_thin.vs.find(id=i).index
    except: # the node we want has been deleted
        return False

    u = G_thin.bfs(i_thin,mode='IN')[0]

    for j in set(u):
        if G_thin.vs[j]['id'] in t:
            return True

    return False

def random_thinning(G,rho):
    return G.induced_subgraph((np.random.random(G.vcount()) <= rho).nonzero()[0].tolist())

def random_failure_reachability(G,rho=np.arange(0,1,.1)):
    avg = []
    for r in rho:
        print(r)
        reachable = []
        for i in G.vs(_outdegree_eq = 0):
            print(len(reachable)/len(G.vs(_outdegree_eq = 0)))
            G_thin = random_thinning(G,r)
            reachable.append(some_terminal_suppliers_reachable(i.index,G,G_thin))
        avg.append(np.mean(reachable))

    plt.scatter(rho,avg)
    plt.plot(rho,rho)
    plt.xlabel("Percent of solvent firms")
    plt.ylabel("Percent of medical supply firms able to reach at least one terminal supplier")
    plt.title("Medical supply chain resilience under random firm failure")

    return avg

def no_china_us_reachability(G):
    print("Removing all US-China supply chain links")
    from copy import deepcopy
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(_between = [G_thin.vs(country = 'United States'), G_thin.vs(country='China')]))
    G_thin.delete_edges(G_thin.es.select(_between = [G_thin.vs(country = 'China'), G_thin.vs(country='US')]))
    print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    reachable = []
    reachable_us = []
    reachable_ch = []
    for i in G.vs(_outdegree_eq = 0):
        #print(len(reachable)/len(G.vs(_outdegree_eq = 0)))
        ans = some_terminal_suppliers_reachable(i.index,G,G_thin)
        reachable.append(ans)
        if i['country'] == 'United States':
            reachable_us.append(ans)
        elif i['country'] == 'China':
            reachable_ch.append(ans)
    avg = np.mean(reachable)
    print("Percent of medical supply firms unable cut off from all terminal suppliers: " + str(1-avg))

    avg_us = np.mean(reachable_us)
    avg_ch = np.mean(reachable_ch)
    print("Percent of US medical supply firms cut off from all terminal suppliers: " + str(1-avg_us))
    print("Percent of Chinese medical supply firms cut off from all terminal suppliers: " + str(1-avg_ch))

    return avg
