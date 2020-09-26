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

def get_firm_df():
    df = get_df()

    firm_df = pd.DataFrame(dict(ID=df['Source'],name=df['Source NAME'],industry=df['Source Industry'],country=df['Source Country']))
    firm_df = firm_df.append(pd.DataFrame(dict(ID=df['Target'],name=df['Target NAME'],industry=df['Target Industry'],country=df['Target Country'])))
    firm_df = firm_df.drop_duplicates()
    firm_df = firm_df.set_index('ID')

    return firm_df

def get_edge_df():
    df = get_df()
    return df[['Source','Target','Tier']]

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
    G.es['tier'] = get_edge_df().Tier
    G.simplify(loops=False, combine_edges='min') # only keep the smaller tier value
    firm_df=get_firm_df()
    G.vs['name']=firm_df['name']
    G.vs['industry']=firm_df['industry']
    G.vs['country']=firm_df['country']
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

def get_u(i,G):
    try:
        i_thin = G.vs.find(id=i).index
    except: # the node we want has been deleted
        return False

    u = G.bfs(i_thin,mode='IN')
    u = u[0][:u[1][-1]] # remove trailing zeros
    return u

# i is an integer, the index of the node in G
def some_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    if set(u) & set(t): # set intersection
        return True
    return False

def all_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    if set(t).issubset(u):
        return True
    return False

def random_thinning(G,rho):
    return G.induced_subgraph((np.random.random(G.vcount()) <= rho).nonzero()[0].tolist())

def random_failure_reachability(G,rho=np.arange(0,1,.1)):

    med_suppliers = [x.target_vertex for x in G.es(tier = 1)]
    #med_suppliers = G.vs(_outdegree_eq = 0)

    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    avg = []
    all_avg = []
    for r in rho:
        print(r)
        G_thin = random_thinning(G,r)
        u = [get_u(i.index,G_thin) for i in med_suppliers]
        reachable = []
        all_reachable = []
        for j in range(len(med_suppliers)):
            i = med_suppliers[j]
            #print(len(reachable)/len(med_suppliers))
            if u[j]:
                reachable.append(some_terminal_suppliers_reachable(i.index,G,G_thin,t[j],u[j]))
                all_reachable.append(all_terminal_suppliers_reachable(i.index,G,G_thin,t[j],u[j]))
            else:
                reachable.append(False)
                all_reachable.append(False)
        avg.append(np.mean(reachable))
        all_avg.append(np.mean(all_reachable))

    import matplotlib.pyplot as plt
    plt.scatter(rho,avg)
    plt.scatter(rho,all_avg)
    plt.plot(rho,rho)
    plt.xlabel("Percent of remaining firms")
    plt.ylabel("Percent of medical supply firms")
    plt.title("Medical supply chain resilience under random firm failure")
    plt.legend(['Expected firms remaining', 'Some end suppliers accessible','All end suppliers accessible'])

    return (avg,all_avg)

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
    reachable_all = []
    reachable_us_all = []
    reachable_ch_all = []
    med_suppliers = [x.target_vertex for x in G.es(tier = 1)]
    for i in med_suppliers:
        print(len(reachable)/len(med_suppliers))
        reachable.append(some_terminal_suppliers_reachable(i.index,G,G_thin))
        reachable_all.append(all_terminal_suppliers_reachable(i.index,G,G_thin))
        if i['country'] == 'United States':
            reachable_us.append(ans)
            reachable_us_all.append(ans)
        elif i['country'] == 'China':
            reachable_ch.append(ans)
            reachable_ch_all.append(ans)
    avg = np.mean(reachable)
    avg_all = np.mean(reachable_all)
    print("Percent of medical supply firms cut off from all terminal suppliers: " + str(1-avg))
    print("Percent of medical supply firms cut off from some terminal suppliers: " + str(1-avg_all))

    avg_us = np.mean(reachable_us)
    avg_ch = np.mean(reachable_ch)
    avg_us_all = np.mean(reachable_us_all)
    avg_ch_all = np.mean(reachable_ch_all)
    print("Percent of US medical supply firms cut off from all terminal suppliers: " + str(1-avg_us))
    print("Percent of Chinese medical supply firms cut off from all terminal suppliers: " + str(1-avg_ch))
    print("Percent of US medical supply firms cut off from some terminal suppliers: " + str(1-avg_us_all))
    print("Percent of Chinese medical supply firms cut off from some terminal suppliers: " + str(1-avg_ch_all))

    return avg,avg_all

def close_all_borders:
    print("Removing all US-China supply chain links")
    from copy import deepcopy
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(_between = [G_thin.vs(country = 'United States'), G_thin.vs(country='China')]))
    print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    reachable = []
    reachable_us = []
    reachable_ch = []
    reachable_all = []
    reachable_us_all = []
    reachable_ch_all = []
    med_suppliers = [x.target_vertex for x in G.es(tier = 1)]
    for i in med_suppliers:
        print(len(reachable)/len(med_suppliers))
        reachable.append(some_terminal_suppliers_reachable(i.index,G,G_thin))
        reachable_all.append(all_terminal_suppliers_reachable(i.index,G,G_thin))
        if i['country'] == 'United States':
            reachable_us.append(ans)
            reachable_us_all.append(ans)
        elif i['country'] == 'China':
            reachable_ch.append(ans)
            reachable_ch_all.append(ans)
    avg = np.mean(reachable)
    avg_all = np.mean(reachable_all)
    print("Percent of medical supply firms cut off from all terminal suppliers: " + str(1-avg))
    print("Percent of medical supply firms cut off from some terminal suppliers: " + str(1-avg_all))

    avg_us = np.mean(reachable_us)
    avg_ch = np.mean(reachable_ch)
    avg_us_all = np.mean(reachable_us_all)
    avg_ch_all = np.mean(reachable_ch_all)
    print("Percent of US medical supply firms cut off from all terminal suppliers: " + str(1-avg_us))
    print("Percent of Chinese medical supply firms cut off from all terminal suppliers: " + str(1-avg_ch))
    print("Percent of US medical supply firms cut off from some terminal suppliers: " + str(1-avg_us_all))
    print("Percent of Chinese medical supply firms cut off from some terminal suppliers: " + str(1-avg_ch_all))

    return avg,avg_all

# TODO: percent of terminal suppliers unreachable
# TODO: all international borders close, with results by country
