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
    firm_df = firm_df.drop_duplicates() # does this always keep the first on? wondering about an index error
    firm_df = firm_df.set_index('ID')

    return firm_df

def get_edge_df():
    df = get_df()
    return df[['Source','Target','Tier']]

def edges():
    df=get_edge_df()
    e,i = pd.factorize(df.Source.append(df.Target,ignore_index=True)) # not guaranteed to respect order of appearance, but it does in practice
    
    m = e.size//2
    
    ee=np.zeros((m,2),dtype=int)
    ee[:,0]=e[:m]
    ee[:,1]=e[m:]
    e=ee
    return e,i

def undirected_igraph(e=None,giant=True):
    if e is None: e,_=edges() # can't have this as a default argument, since these are evaluated at import time, and it is time consuming
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

def directed_igraph(giant=False,no_software=True):

    firm_df = get_firm_df()
    edge_df = get_edge_df()
    G = ig.Graph(directed=True)
    G.add_vertices(firm_df.index)
    G.vs['firm name']=firm_df['name']
    G.vs['industry']=firm_df['industry']
    G.vs['country']=firm_df['country']
    G.add_edges(edge_df[['Source','Target']].itertuples(index=False))
    G.es['tier'] = edge_df.Tier
    #G.simplify(loops=False, combine_edges='min') # use min to keep smaller tier value. probably unnecessary

    if no_software:
        G=G.induced_subgraph(
                [x.index for x in 
                    G.vs(lambda x : x['industry'] not in 
                        ['Application Software', 'IT Consulting and Other Services', 'Systems Software', 'Advertising', 'Movies and Entertainment', 'Interactive Home Entertainment'])])

    if giant:
        G=G.components().giant()

    G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs

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

def percent_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    return len(set(t) & set(u))/len(set(t))

def random_thinning(G,rho):
    return G.induced_subgraph((np.random.random(G.vcount()) <= rho).nonzero()[0].tolist())

def random_failure_reachability(G,rho=np.arange(0,1,.1)):

    med_suppliers = [x.target_vertex for x in G.es(tier = 1)]
    #med_suppliers = G.vs(_outdegree_eq = 0)

    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    avg = []
    all_avg = []
    per_avg = []
    for r in rho:
        print(r)
        reachable = []
        all_reachable = []
        per_reachable = []

        G_thin = random_thinning(G,r)
        u = [get_u(i.index,G_thin) for i in med_suppliers]
        for j in range(len(med_suppliers)):
            i = med_suppliers[j]
            #print(len(reachable)/len(med_suppliers))
            if u[j]:
                reachable.append(some_terminal_suppliers_reachable(i.index,G,G_thin,t[j],u[j]))
                all_reachable.append(all_terminal_suppliers_reachable(i.index,G,G_thin,t[j],u[j]))
                per_reachable.append(percent_terminal_suppliers_reachable(i.index,G,G_thin,t[j],u[j]))
            else:
                reachable.append(False)
                all_reachable.append(False)
                per_reachable.append(0)
        avg.append(np.mean(reachable))
        all_avg.append(np.mean(all_reachable))
        per_avg.append(np.mean(per_reachable))

    import matplotlib.pyplot as plt
    plt.scatter(rho,avg)
    plt.scatter(rho,all_avg)
    plt.scatter(rho,per_avg)
    plt.plot(rho,rho)
    plt.xlabel("Percent of remaining firms")
    plt.ylabel("Percent of medical supply firms")
    plt.title("Medical supply chain resilience under random firm failure")
    plt.legend(['Expected firms remaining', 'Some end suppliers accessible','All end suppliers accessible','Avg. percent end suppliers reachable'])

    return (avg,all_avg,per_avg)

def no_china_us_reachability(G,include_taiwan_hong_kong=False):
    print("Removing all US-China supply chain links")

    # Get medical suppliers
    firms = G.vs
    firms['is_med'] = [False] * G.vcount()
    for x in G.es(tier = 1): x.target_vertex['is_med'] = True

    # define china
    china=['China']
    if include_taiwan_hong_kong:
        print("Including Taiwan and Hong Kong in China")
        china += ['Hong Kong','Taiwan']

    # find chinese/us firms
    is_ch = lambda x : type(x['country'] == str) and (x['country'] in china)
    is_us = lambda x : x['country'] == 'United States'

    # thin graph
    from copy import deepcopy
    G_thin = deepcopy(G)
    G_thin.delete_edges(
            G_thin.es.select(
                _between = [
                    G_thin.vs(is_ch), 
                    G_thin.vs(is_us)]))
    print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    # compute reachability statistics
    na = 'Not applicable'
    firms['reachable_some'] = [(
        some_terminal_suppliers_reachable(i.index,G,G_thin) if i['is_med'] 
        else na)
        for i in firms]
    firms['reachable_all'] = [(
        all_terminal_suppliers_reachable( i.index,G,G_thin) if i['is_med'] 
        else na)
        for i in firms]

    # print outputs
    med_firms = firms(is_med = True)
    percent_reachable_some = len(med_firms(reachable_some = True)) / len(med_firms)
    percent_reachable_all  = len(med_firms(reachable_all  = True)) / len(med_firms)
    print("Percent of medical supply firms cut off from all terminal suppliers: " + str(1-percent_reachable_some))
    print("Percent of medical supply firms cut off from some terminal suppliers: " + str(1-percent_reachable_all))

    us_med_firms = med_firms(lambda x : x['is_med'] and is_us(x))
    ch_med_firms = med_firms(lambda x : x['is_med'] and is_ch(x))
    percent_reachable_some_us = len(us_med_firms(reachable_some = True)) / len(us_med_firms)
    percent_reachable_some_ch = len(ch_med_firms(reachable_some = True)) / len(ch_med_firms)
    percent_reachable_all_us  = len(us_med_firms(reachable_all  = True)) / len(us_med_firms)
    percent_reachable_all_ch  = len(ch_med_firms(reachable_all  = True)) / len(ch_med_firms)
    print("Percent of US medical supply firms cut off from all terminal suppliers: " + str(1-percent_reachable_some_us))
    print("Percent of Chinese medical supply firms cut off from all terminal suppliers: " + str(1-percent_reachable_some_ch))
    print("Percent of US medical supply firms cut off from some terminal suppliers: " + str(1-percent_reachable_all_us))
    print("Percent of Chinese medical supply firms cut off from some terminal suppliers: " + str(1-percent_reachable_all_ch))

def close_all_borders(G):
    print("Removing all international supply chain links")
    from copy import deepcopy
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(lambda e : e.source_vertex['country'] != e.target_vertex['country']))
    print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = [x.target_vertex for x in G.es(tier = 1)]

    reachable = []
    reachable_all = []
    reachable_per = []
    country_reachability = {s['country']:[] for s in set(med_suppliers)}
    country_reachability_all = {s['country']:[] for s in set(med_suppliers)}
    for i in med_suppliers:
        print(len(reachable)/len(med_suppliers))
        s = some_terminal_suppliers_reachable(i.index,G,G_thin)
        reachable.append(s)
        country_reachability[i['country']].append(s)
        
        a = all_terminal_suppliers_reachable(i.index,G,G_thin)
        reachable_all.append(all_terminal_suppliers_reachable(i.index,G,G_thin))
        country_reachability_all[i['country']].append(a)

        p = percent_terminal_suppliers_reachable(i.index,G,G_thin)
        reachable_per.append(p)
    avg = np.mean(reachable)
    avg_all = np.mean(reachable_all)
    avg_per = np.mean(reachable_per)
    print("Percent of medical supply firms cut off from all terminal suppliers: " + str(1-avg))
    print("Percent of medical supply firms cut off from some terminal suppliers: " + str(1-avg_all))
    print("Avg. percent of terminal suppliers reachable by medical supply firms: " + str(1-avg_per))

    import math
    for c,l in country_reachability.items():
        print("Percent of medical supply firms in " + str(c)  + ' cut off from all terminal suppliers: ' + str(1-np.mean(l))) # need str for nan
    for c,l in country_reachability_all.items():
        print("Percent of medical supply firms in " + str(c) + ' cut off from some terminal suppliers: ' + str(1-np.mean(l)))

    return avg,avg_all,avg_per

# TODO Redo random plot for smaller tier counts
# TODO Figure out which nodes are most important under random deletion
# TODO Random deletion analysis at the “industry level” and “country level”
