import pandas as pd
import igraph as ig
import numpy as np
import ipyparallel
import matplotlib.pyplot as plt
import pickle
import logging
logging.basicConfig(filename='.med.log',level=logging.DEBUG,format='%(levelname)s:%(message)s',filemode='w')

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
    firm_df = firm_df.drop_duplicates() # does this always keep the first one? wondering about an index error
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

def directed_igraph(giant=False,no_software=True,reverse_direction=False):

    firm_df = get_firm_df()
    edge_df = get_edge_df()
    G = ig.Graph(directed=True)
    G.add_vertices(firm_df.index)
    G.vs['firm name']=firm_df['name']
    G.vs['industry']=firm_df['industry']
    G.vs['country']=firm_df['country']

    if reverse_direction:
        G.add_edges(edge_df[['Target','Source']].itertuples(index=False))
    else:
        G.add_edges(edge_df[['Source','Target']].itertuples(index=False))

    G.es['tier'] = edge_df.Tier
    #G.simplify(loops=False, combine_edges='min') # use min to keep smaller tier value. probably unnecessary

    med_suppliers = [x.target_vertex.index for x in G.es(tier = 1)]
    G.vs['is_medical'] = [i in med_suppliers for i in range(G.vcount())]

    if no_software:
        G=G.induced_subgraph(
                [x.index for x in 
                    G.vs(lambda x : x['industry'] not in 
                        ['Application Software', 'IT Consulting and Other Services', 'Systems Software', 'Advertising', 'Movies and Entertainment', 'Interactive Home Entertainment'])])

    if giant:
        G=G.components().giant()

    G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs

    return G

import itertools
def get_terminal_suppliers(i,G):

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
    if type(i) is not int:
        raise TypeError

    try:
        i_thin = G.vs.find(id=i).index
    except: # the node we want has been deleted
        return False

    u = G.bfs(i_thin,mode='IN')
    u = u[0][:u[1][-1]] # remove trailing zeros

    u = [G.vs[i]['id'] for i in u] 
    return u

def is_increasing(l):
    if len(l) < 2:
        return True
    for i in range(len(l)-1):
        if l[i] > l[i+1]:
            return False
    return True

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

def all_terminal_suppliers_exist(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)

    return len(set(G_thin.vs['id']) & set(t)) == len(set(t))

def is_subgraph(G_thin,G):

    # check nodes
    for i in G_thin.vs['id']:
        if i not in G.vs['id']:
            return False

    # check edges
    for e in G_thin.es:
        s=e.source_vertex['id']
        t=e.target_vertex['id']
        sv = G.vs(id = s)[0]
        tv = G.vs(id = t)[0]
        if not tv in sv.successors():
            return False

    return True

import random
def random_thinning(G,rho,failure_type='firm'):

    if failure_type == 'firm':
        return G.induced_subgraph((np.random.random(G.vcount()) <= rho).nonzero()[0].tolist())

    uniques = set([x for x in G.vs[failure_type] if type(x) is not float])
    uniques.add('nan')
    keep_uniques = random.sample(uniques,k=round(rho*len(uniques)))
    return G.induced_subgraph(G.vs(lambda x : str(x[failure_type]) in keep_uniques))

def target_by_attribute(G,attr):
    sorted_attr_inds = sorted(range(G.vcount()), key=G.vs[attr].__getitem__)
    #sorted_attr_inds_country = sorted(set(G['country']), 
            #key=lambda x : sum(G.vs(country = x)[attr])) # TODO: check if na is correectly handled

    def targeted(G,r,failure_type='firm'):
        if failure_type != 'firm': raise NotImplementedError
        return G.induced_subgraph(
                sorted_attr_inds[:int( (G.vcount()-1) * r )])
    return targeted

def get_degree_attack(G):
    G.vs['degree'] = G.degree(range(G.vcount()))
    return target_by_attribute(G,'degree')

def get_pagerank_attack(G,transpose=True):

    if transpose:
        Gt=deepcopy(G)
        tier = dict(tier=Gt.es['tier'])
        edges = [tuple(reversed(e.tuple)) for e in G.es]
        Gt.delete_edges(None)
        Gt.add_edges(edges,tier)
        G.vs['pagerank']=Gt.pagerank()
        del Gt # don't accidentally reference later
    else:
        G.vs['pagerank']=G.pagerank()

    return target_by_attribute(G,'pagerank')

def _random_failure_reachability_inner(r,G,med_suppliers,ts,failure_type='firm',callbacks=[],targeted=False):

    G_thin = targeted(G,r,failure_type=failure_type) if targeted else random_thinning(G,r,failure_type=failure_type)

    res = []
    us = [get_u(i,G_thin) for i in med_suppliers]
    for cb in callbacks:
        sample = [cb(med_suppliers,G,G_thin,t,u) for i,t,u in zip(med_suppliers,ts,us)]
        res.append(np.mean(sample))
    return res

def _random_failure_reachability(rho,G,med_suppliers,ts,failure_type='firm',callbacks=[],targeted=False,parallel=False):
    l=logging.getLogger('.med.log')
    l.setLevel(logging.INFO)
    l.addHandler(logging.FileHandler('.med.log'))
    logging.info('Starting _random_failure_reachability')
    avgs = []
    if parallel:
        avgs = ipyparallel.Client().load_balanced_view().map(_random_failure_reachability_inner,
            rho,*list(zip(*[[G,med_suppliers,ts,failure_type,callbacks,targeted]]*len(rho))))
    else:
        for r in rho:
            print(r)
            avgs.append(_random_failure_reachability_inner(r,G,med_suppliers,ts,failure_type=failure_type,callbacks=callbacks,targeted=targeted))
    return list(zip(*avgs))

import seaborn as sns
def random_failure_plot(rho,callbacks,avgs,plot_title='Medical supply chain resilience under random firm failures'):

    rho = np.tile(rho,len(avgs[0])//len(rho))
    ax=[]
    ax = [sns.lineplot(x=rho,y=avg,label=cb.description) for avg,cb in zip(avgs,callbacks)]
    ax.append(sns.lineplot(x=rho,y=rho,label='Expected firms remaining'))
    ax[0].set(xlabel='Percent of remaining firms',
            ylabel='Percent of firms',
            title='Medical supply chain resilience under random firm failures')
    plt.legend()

class callback:
    def __init__(self,function,default,description):
        self.function = function
        self.default = default
        self.description = description

    def __call__(self,med_suppliers,G,G_thin,t=None,u=None):
        return self.function(med_suppliers,G,G_thin,t,u) if u or callable(self.default) \
            else self.default

callbacks = [
    callback(some_terminal_suppliers_reachable,False,'Some end suppliers accessible'),
    callback(all_terminal_suppliers_reachable,False,'All end suppliers accessible'),
    callback(percent_terminal_suppliers_reachable,0,'Avg. percent end supplier reachable'),
    callback(all_terminal_suppliers_exist,all_terminal_suppliers_exist,'All end suppliers surviving')]

def random_failure_reachability(G,
        rho=np.linspace(0,1,10),
        tiers=5,
        plot=True,
        repeats=1,
        failure_type='firm',
        use_cached_t=False,
        targeted=False,
        parallel='auto',
        plot_title='Medical supply chain resilience under random firm failures',
        callbacks=callbacks):

    if parallel == 'auto':
        parallel = 'repeat' if repeats > 1 else 'rho'

    if tiers < 5:
        G = deepcopy(G)
        G.delete_edges(G.es(tier_ge = tiers+1))

    med_suppliers = [x.target_vertex.index for x in G.es(tier = 1)]

    if use_cached_t:
        t=[]
        with open('.cached_t','rb') as f: t=pickle.load(f)
    else:
        t = [get_terminal_suppliers(i,G) for i in med_suppliers]
        with open('.cached_t','wb') as f: pickle.dump(t,f)

    args = [rho,G,med_suppliers,t,failure_type,callbacks,targeted]
    if parallel == 'repeat':
        avgs = ipyparallel.Client().load_balanced_view().map(
                _random_failure_reachability,
                *list(zip(*[args]*repeats)))
    elif parallel == 'rho':
        avgs = [_random_failure_reachability(*args,parallel=True)]
    else:
        avgs = [_random_failure_reachability(*args) for _ in range(repeats)]
    a = zip(*avgs)
    avgs = [sum(b,()) for b in a]

    if plot:
        random_failure_plot(rho,callbacks,avgs,plot_title=plot_title)

    return avgs

def compare_tiers_random(G, rho=np.linspace(0,1,101), repeats=25, plot=True):
    
    trange = range(2,6)
    avgs = [random_failure_reachability(
        G,
        rho=rho,
        tiers=tiers,
        plot=False,
        use_cached_t = False,
        callbacks=(callbacks[2],),
        repeats=repeats)[0] for tiers in trange]

    if plot:
        rho = np.tile(rho,len(avgs[0])//len(rho))
        ax=[]
        ax = [sns.lineplot(x=rho,y=avg,label=str(i) + ' tiers') for i,avg in zip(trange,avgs)]
        ax.append(sns.lineplot(x=rho,y=rho,label='Expected firms remaining'))
        ax[0].set(xlabel='Percent remaining firms',
                ylabel='Percent end suppliers reachable',
                title='Medical supply chain resilience with different tier counts')
        plt.legend()

    return avgs

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

def get_med_suppliers(G):
    return [x.target_vertex for x in G.es(tier = 1)]

from copy import deepcopy
import math
def close_all_borders(G):
    print("Removing all international supply chain links")
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(lambda e : e.source_vertex['country'] != e.target_vertex['country']))
    print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = get_med_suppliers(G)

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

    for c,l in country_reachability.items():
        print("Percent of medical supply firms in " + str(c)  + ' cut off from all terminal suppliers: ' + str(1-np.mean(l))) # need str for nan
    for c,l in country_reachability_all.items():
        print("Percent of medical supply firms in " + str(c) + ' cut off from some terminal suppliers: ' + str(1-np.mean(l)))

    return avg,avg_all,avg_per

def industry_deletion_effects(G):
    print("Removing each industry one at a time")

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    res = dict()
    for ind in set(G.vs['industry']):
        if type(ind) is not float:
            print(ind)
            G_thin = deepcopy(G)
            G_thin.delete_vertices(G.vs(industry=ind))
            u = [get_u(i.index,G_thin) for i in med_suppliers]
            print("Deleting " + str(G.vcount() - G_thin.vcount()) + " vertices")
            res[ind] = [np.mean([cb(None,G,G_thin,tt,uu) for tt,uu in zip(t,u)]) for cb in callbacks]
            print(res[ind])

    print('Statistics in order: ')
    for cb in callbacks:
        print(cb.description)
    for key,val in res.items():
        print(key + ": " + ''.join([str(i) + ' ' for i in val]))

    return res

