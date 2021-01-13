import pandas as pd
import igraph as ig
import numpy as np
import ipyparallel
import matplotlib.pyplot as plt
import pickle
import logging
import os
from copy import deepcopy
import math
import itertools
import random
import seaborn as sns
from functools import partial
import matplotlib
#logging.basicConfig(filename='.med.log',level=logging.DEBUG,format='%(levelname)s:%(message)s',filemode='w')

def get_df(cached=True,save=True):
    if cached:
        df = pd.read_hdf('kayvan_data.h5')
    else:
        file_name='dat/Backward SC 2020-09-19 Tiers 1,2,3,4,5 Supplier Only In one Column-Size-Type v2.xlsx'
        df = pd.read_excel(file_name,sheet_name="Tier 1-5")
        df = df.drop('Type',axis=1)
        df = df.drop_duplicates(ignore_index=True)
        if save:
            df.to_hdf('kayvan_data.h5',key='df')
    return df

def get_firm_df():
    df = get_df()

    firm_df = pd.DataFrame(dict(
        ID=df['Source'],
        name=df['Source NAME'],
        industry=df['Source Industry'],
        country=df['Source Country'],
        market_cap=df['SourceSizeMktCap'],
        revenue=df['SourceSizeRevenue'],
        employees=df['SourceSizeEmployeesGlobal'],
        private=df['SourceType']))
    firm_df = firm_df.append(pd.DataFrame(dict(
        ID=df['Target'],
        name=df['Target NAME'],
        industry=df['Target Industry'],
        country=df['Target Country'],
        market_cap=df['TargetSizeMktCap'],
        revenue=df['TargetSizeRevenue'],
        employees=df['TargetSizeEmployeesGlobal'],
        private=df['TargetType'])))

    for col in ['country', 'industry']:
        firm_df[col] = firm_df[col].astype(str) # make NaNs into string nan
    firm_df.private = firm_df.private == 'Private Company'
    firm_df = firm_df.drop_duplicates() # does this always keep the first one? wondering about an index error
    firm_df = firm_df.set_index('ID')
    firm_df['country-industry'] = firm_df['country'] + ' '  + firm_df['industry']
    firm_df.loc[firm_df['employees'] == '(Invalid Identifier)'] = math.nan

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

def directed_igraph(giant=True,no_software=True,reverse_direction=False,include_private=True,cut_low_terminal_suppliers=True):

    firm_df = get_firm_df()
    edge_df = get_edge_df()
    G = ig.Graph(directed=True)
    G.add_vertices(firm_df.index)
    G.vs['firm name']=firm_df['name']
    for attr in ['industry', 'country', 'country-industry','market_cap','revenue','employees','private']:
        G.vs[attr]=firm_df[attr]

    if reverse_direction:
        G.add_edges(edge_df[['Target','Source']].itertuples(index=False))
    else:
        G.add_edges(edge_df[['Source','Target']].itertuples(index=False))

    G.es['tier'] = edge_df.Tier
    #G.simplify(loops=False, combine_edges='min') # use min to keep smaller tier value. probably unnecessary

    G.vs['tier_set'] = [set([e['tier'] for e in n.all_edges()])
        for n in G.vs]

    if no_software:
        G=G.induced_subgraph(
                [x.index for x in 
                    G.vs(lambda x : x['industry'] not in 
                        ['Application Software', 'IT Consulting and Other Services', 'Systems Software', 'Advertising', 'Movies and Entertainment', 'Interactive Home Entertainment'])])

    if giant:
        G=G.components(mode='WEAK').giant()

    G.vs['id'] = list(range(G.vcount())) # helps when passing to subgraphs
    med_suppliers = get_med_suppliers(G)
    G.vs['is_medical'] = [i in med_suppliers for i in range(G.vcount())]

    if cut_low_terminal_suppliers:
        t = [get_terminal_suppliers(i,G) for i in med_suppliers]
        to_delete = [m for m,tt in zip(med_suppliers,t) if len(tt) < 31] # 31 because there is a jump in len(tt) here
        G.delete_vertices(to_delete)
        G.vs['id'] = list(range(G.vcount())) # recalculate

    return G

def get_terminal_suppliers(i,G):
    if type(i) is ig.Vertex:
        i = i.index

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

    return set(terminal_nodes)

def get_u(i,G_thin,med_suppliers_thin=None):
    if type(i) is ig.Vertex:
        i = i.index

    try:
        i_thin = med_suppliers_thin[i] if med_suppliers_thin else G_thin.vs.find(id=i).index
    except: # the node we want has been deleted
        return set()
        #return False

    u = G_thin.bfs(i_thin,mode='IN')
    u = u[0][:u[1][-1]] # remove trailing zeros

    ids = G_thin.vs['id']
    return {ids[i] for i in u}

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

    if u & t: # set intersection
        return True
    return False
some_terminal_suppliers_reachable.description='Some end suppliers accessible'

def all_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    return t.issubset(u)
all_terminal_suppliers_reachable.description='All end suppliers accessible'

def percent_terminal_suppliers_reachable(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)
    if u is None: u = get_u(i,G_thin)

    return len(t & u)/len(t)
percent_terminal_suppliers_reachable.description='Avg. percent end supplier reachable'

def all_terminal_suppliers_exist(i,G,G_thin,t=None,u=None):
    if t is None: t = get_terminal_suppliers(i,G)

    return t.issubset(G_thin.vs['id'])
all_terminal_suppliers_exist.description='All end suppliers surviving'

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

def random_thinning_factory(G):
    firm_rands = np.random.random(G.vcount())

    uniques = dict()
    perm = dict()
    for failure_scale in ['country', 'industry', 'country-industry']:
        uniques[failure_scale] = list(set(G.vs[failure_scale]))
        perm[failure_scale] = uniques[failure_scale]
        random.shuffle(perm[failure_scale])

    def attack(G,rho,failure_scale='firm'):
        if failure_scale == 'firm':
            return G.induced_subgraph((firm_rands <= rho).nonzero()[0].tolist())
        else:
            keep_uniques = perm[failure_scale][:round(rho*len(uniques[failure_scale]))]
            return G.induced_subgraph(G.vs(lambda x : x[failure_scale] in keep_uniques))
    attack.description = 'random'

    return attack
random_thinning_factory.description = 'random'

def target_by_attribute(G,attr):

    sorted_attr_inds=dict()
    sorted_attr_inds['firm'] = sorted(range(G.vcount()), key=G.vs[attr].__getitem__)
    for failure_scale in ['country', 'industry', 'country-industry']:
        sorted_attr_inds[failure_scale]  = sorted(set(G.vs[failure_scale ]), key=lambda x : sum(G.vs(lambda v : v[failure_scale] == x)[attr]))

    def targeted(G,r,failure_scale='firm'):
        to_keep = sorted_attr_inds[failure_scale][:int( (len(sorted_attr_inds[failure_scale])-1) * r )]
        if failure_scale == 'firm':
            return G.induced_subgraph(to_keep)
        else:
            return G.induced_subgraph(G.vs(lambda x : str(x[failure_scale]) in to_keep))

    targeted.description = attr+'-targeted'
            
    return targeted

def get_employee_attack(G):
    G=deepcopy(G)
    G.vs['employees_imputed'] = [math.isnan(x) for x in G.vs['employees']]
    size_dist_private = np.array([x['employees'] for x in G.vs if x['private'] and not math.isnan(x['employees'])])
    imputed_size = np.random.choice(size_dist_private,len(G.vs(employees_imputed_eq = True)))
    for v,s in zip(G.vs(employees_imputed_eq = True),imputed_size):
        v['employees'] = s
    return target_by_attribute(G,'employees')
get_employee_attack.description = 'employee-targeted'

def get_degree_attack(G):
    G.vs['degree'] = G.degree(range(G.vcount()))
    return target_by_attribute(G,'degree')
get_degree_attack.description = 'degree-targeted'

def get_pagerank_attack(G,transpose=True):

    attrname = 'pagerank-of-transpose' if transpose else 'pagerank'
    try:
        G[attrname]
    except:
        if transpose:
            Gt=deepcopy(G)
            tier = dict(tier=Gt.es['tier'])
            edges = [tuple(reversed(e.tuple)) for e in G.es]
            Gt.delete_edges(None)
            Gt.add_edges(edges,tier)
            G.vs[attrname]=Gt.pagerank()
            del Gt # don't accidentally reference later
        else:
            G.vs[attrname]=G.pagerank()

    return target_by_attribute(G,attrname)
get_pagerank_attack.description='pagerank'

def get_null_attack(G):
    def null_attack(G,r,failure_scale='firm'):
        return G
    return null_attack
get_null_attack.description='null-targeted'

dv = ipyparallel.Client()[:] # This should be global (or a singleton) to avoid an error with too many files open https://github.com/ipython/ipython/issues/6039
dv.block=False
dv.use_dill()

callbacks = [
some_terminal_suppliers_reachable,
percent_terminal_suppliers_reachable,
]

def failure_reachability_single(r,G,med_suppliers=False,ts=False,failure_scale='firm',callbacks=callbacks,targeted=False):

    if not med_suppliers:
        med_suppliers=get_med_suppliers(G)
    if not ts:
        ts = [get_terminal_suppliers(i,G) for i in med_suppliers]
    if not targeted:
        targeted=random_thinning_factory(G)

    G_thin = targeted(G,r,failure_scale=failure_scale)
    med_suppliers_thin = {i_thin['id']:i_thin.index for i_thin in G_thin.vs if i_thin['id'] in med_suppliers}

    res = []
    us = [get_u(i,G_thin, med_suppliers_thin) for i in med_suppliers]
    for cb in callbacks:
        sample = [cb(med_suppliers,G,G_thin,t,u) for i,t,u in zip(med_suppliers,ts,us)]
        res.append(np.mean(sample))
    return res

def failure_reachability_sweep(rho,G,med_suppliers=False,ts=False,failure_scale='firm',callbacks=callbacks,targeted_factory=random_thinning_factory,parallel=False):

    if not med_suppliers:
        med_suppliers=get_med_suppliers(G)
    if not ts:
        ts = [get_terminal_suppliers(i,G) for i in med_suppliers]

    avgs = []
    if parallel:
        avgs = dv.map(failure_reachability_single,
            rho,*list(zip(*[[G,med_suppliers,ts,failure_scale,callbacks,targeted_factory(G)]]*len(rho))))
    else:
        targeted=targeted_factory(G)
        for r in rho:
            print(r)
            avgs.append(failure_reachability_single(r,G,med_suppliers,ts,failure_scale=failure_scale,callbacks=callbacks,targeted=targeted))
    return list(zip(*avgs))

def failure_plot(rho,callbacks,avgs,plot_title='Medical supply chain resilience under random firm failures',save_only=False,filename=None,xlabel='Percent firms remaining'):

    rho = np.tile(rho,len(avgs[0])//len(rho))
    ax=[]
    ax = [sns.lineplot(x=rho,y=avg,label=cb.description) for avg,cb in zip(avgs,callbacks)]
    ax.append(sns.lineplot(x=rho,y=rho,label=xlabel))
    ax[0].set(xlabel=xlabel,
            ylabel='Percent of firms',
            title=plot_title)
    plt.legend()

    if save_only:
        plt.savefig(filename)

def failure_reachability(G,
        rho=np.linspace(0,1,10),
        tiers=5,
        plot=True,
        save_only=False,
        repeats=1,
        failure_scale='firm',
        use_cached_t=False,
        targeted_factory=random_thinning_factory,
        parallel='auto',
        callbacks=callbacks,
        G_has_no_software_flag=None,
        prefix='',
        med_suppliers=None):

    if parallel == 'auto' or parallel == True:
        parallel = 'repeat' if repeats > 1 else 'rho'

    if tiers < 5:
        G = deepcopy(G)
        G.delete_edges(G.es(tier_ge = tiers+1))

    if med_suppliers is None:
        med_suppliers = [i.index for i in get_med_suppliers(G)]

    if use_cached_t:
        t=[]
        with open('.cached_t','rb') as f: t=pickle.load(f)
    else:
        t = [get_terminal_suppliers(i,G) for i in med_suppliers]
        with open('.cached_t','wb') as f: pickle.dump(t,f)

    args = [[rho,G,med_suppliers,t,failure_scale,callbacks,targeted_factory]]*repeats

    if parallel == 'repeat':
        avgs = dv.map(failure_reachability_sweep,
                *list(zip(*args)))
    elif parallel == 'rho':
        avgs = [failure_reachability_sweep(*args[0],parallel=True)]
    else:
        avgs = [failure_reachability_sweep(*args[0]) for _ in range(repeats)]
    a = zip(*avgs)
    avgs = [sum(b,()) for b in a]

    if plot:
        plot_title = targeted_factory.description.title() + ' '\
                + failure_scale + ' failures'\
                + ((' ' + str(tiers) + ' tiers') if tiers < 5 else '')\
                + ((' excluding software firms' if G_has_no_software_flag else ' including software firms') if G_has_no_software_flag is not None else '')
        fname = prefix + '/'\
                + failure_scale\
                + '_' + targeted_factory.description\
                + '_range_' + str(rho[0]) + '_' + str(rho[-1])\
                + '_repeats_' + str(repeats)\
                + ('_tiers_' + str(tiers) if tiers < 5 else '')\
                + (('software_excluded' if G_has_no_software_flag else 'software_included') if G_has_no_software_flag is not None else '')
        xlabel='Percent ' + get_plural(failure_scale) + ' remaining'
        if save_only:
            os.makedirs(os.path.dirname('dat/'  + fname),exist_ok=True)
            with open('dat/' + fname+'.pickle',mode='wb') as f:
                pickle.dump(avgs,f)
        os.makedirs(os.path.dirname('im/'  + fname),exist_ok=True)
        failure_plot(rho,
            callbacks,
            avgs,
            plot_title=plot_title,
            save_only=save_only,
            filename='im/'+fname+'.svg',
            xlabel=xlabel)

    return avgs

def get_plural(x):
    if x=='firm': 
        return 'firms'
    elif x=='country': 
        return 'countries'
    elif x=='industry': 
        return 'industries'
    elif x=='country-industry':
        return 'country-industries'
    else:
        raise NotImplementedError

def compare_tiers_random(G, rho=np.linspace(0,1,101), repeats=25, plot=True,save=True):
    
    trange = range(1,6)
    avgs = [failure_reachability(
        G,
        rho=rho,
        tiers=tiers,
        plot=False,
        callbacks=(percent_terminal_suppliers_reachable,),
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
        if save:
            fname = 'compare_tiers_random'
            with open('dat/' + fname+'.pickle',mode='wb') as f:
                pickle.dump(avgs,f)
            plt.savefig('im/'+fname+'.svg')
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
    else:
        print("Excluding Taiwan and Hong Kong from China")

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
    #print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    typ = [bool, bool, float, bool]
    for cb,t in zip(callbacks,typ):
        reachable[cb.description] = reachable[cb.description].astype(t)

    by_country = reachable.groupby('country').mean().loc[['United States', 'China', 'Taiwan', 'Hong Kong']]

    by_country.to_excel('dat/no_us_china'+ ('incl_taiwan_hk' if include_taiwan_hong_kong else '') + '.xlsx')

    return by_country

def get_med_suppliers(G):
    return list({x.target_vertex for x in G.es(tier = 1)})

def close_all_borders(G):
    #print("Removing all international supply chain links")
    G_thin = deepcopy(G)
    G_thin.delete_edges(G_thin.es.select(lambda e : e.source_vertex['country'] != e.target_vertex['country']))
    #print("Percent of edges deleted: " + str(1-G_thin.ecount()/G.ecount()))

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]
    u = [get_u(i.index,G_thin) for i in med_suppliers]

    reachable = pd.DataFrame([[cb(i.index,G,G_thin,tt,uu) for i,tt,uu in zip(med_suppliers,t,u)] for cb in callbacks],index=[cb.description for cb in callbacks]).transpose()
    reachable['country'] = G.vs([i.index for i in med_suppliers])['country']
    typ = [bool, bool, float, bool]
    for cb,t in zip(callbacks,typ):
        reachable[cb.description] = reachable[cb.description].astype(t)

    by_country = reachable.groupby('country').mean()

    by_country.to_excel('dat/close_all_borders.xlsx')

    return reachable

def industry_deletion_effects(G):
    #print("Removing each industry one at a time")

    med_suppliers = get_med_suppliers(G)
    t = [get_terminal_suppliers(i.index,G) for i in med_suppliers]

    res = dict()
    for ind in set(G.vs['industry']):
        if type(ind) is not float:
            #print(ind)
            G_thin = deepcopy(G)
            G_thin.delete_vertices(G.vs(industry=ind))
            u = [get_u(i.index,G_thin) for i in med_suppliers]
            res[ind] = [np.mean([cb(None,G,G_thin,tt,uu) for tt,uu in zip(t,u)]) for cb in callbacks]
            #print(res[ind])

#    print('Statistics in order: ')
#    for cb in callbacks:
#        print(cb.description)
#    for key,val in res.items():
#        print(key + ": " + ''.join([str(i) + ' ' for i in val]))

    res = pd.DataFrame(res,index=[cb.description for cb in callbacks]).transpose()
    res.to_excel('dat/industry_deletion_effects.xlsx')

    return res

def run_all_simulations(G=None,attacks=None,attack_repeats=None,giant=True):
    if G is None:
        G = directed_igraph(giant=giant)

    old_backend = matplotlib.backends.backend
    matplotlib.use('Agg') # non-interactive

    full_rho = np.linspace(.3,1,71)
    failure_scales = ['firm','country','industry','country-industry']
    if attacks is None:
        attacks = [random_thinning_factory, partial(get_pagerank_attack,transpose=True), partial(get_pagerank_attack, transpose=False), get_employee_attack]
        attacks[1].description = 'pagerank-of-transpose-targeted'
        attacks[2].description = 'pagerank-targeted'
        attack_repeats = [25,1,1,25]

    med_suppliers = [i.index for i in get_med_suppliers(G)]

    # Failure type X attack type
    for rho in [full_rho, np.linspace(.9,1,101), np.linspace(.99,1,101),np.linspace(.999,1,101),np.linspace(.9999,1,101)]:
        for failure_scale in failure_scales:
            for attack,repeats in zip(attacks,attack_repeats):
                print(failure_scale + ' ' + (attack.description if attack else 'random') + ' scale ' + str(rho[0]) + ' ' + str(rho[-1]))
                plt.clf()
                failure_reachability(G,rho=rho, targeted_factory=attack, save_only=True,
                        repeats=repeats, failure_scale=failure_scale,
                        prefix='scales_and_attacks_' + str(rho[0]) + '_' + str(rho[-1]),
                        med_suppliers=med_suppliers)

    # tiers
    tiers = range(1,6)
    for tier in tiers:
        for attack,repeats in zip(attacks,attack_repeats):
            print(str(tier) + ' ' + (attack.description if attack else 'random'))
            plt.clf()
            failure_reachability(G,rho=full_rho,tiers=tier,targeted_factory=attack, save_only=True,
                    repeats=repeats,
                    prefix='tiers',
                    med_suppliers=med_suppliers)
    print('compare_tiers_random')
    plt.clf()
    compare_tiers_random(G,rho=full_rho,repeats=25,plot='save')

    # Software include-exclude
    graphs = (directed_igraph(no_software = True, giant=giant), 
            directed_igraph(no_software = False, giant=giant))
    for G, inclusive  in zip(graphs, (False, True)):
        for attack,repeats in zip(attacks,attack_repeats):
            print('Software ' + ('included' if inclusive else 'excluded') + ' ' + (attack.description if attack else 'random'))
            plt.clf()
            failure_reachability(G,rho=full_rho, targeted_factory=attack, save_only=True,
                    repeats=repeats, failure_scale='firm',
                    G_has_no_software_flag = (not inclusive),
                    prefix='software_compare')

    # Country-based and international
    print('no_china_us_reachability')
    no_china_us_reachability(G)
    print('no_china_us_reachability')
    no_china_us_reachability(G,include_taiwan_hong_kong=False)
    print('close_all_borders')
    close_all_borders(G)
    print('industry_deletion_effects')
    industry_deletion_effects(G)

    matplotlib.use(old_backend) # non-interactive
