import powerlaw
import pandas as pd
import igraph as ig
import numpy as np
import ipyparallel as ipp
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
import tables # pytables needed to write to hdf

# User-set parameters

max_tiers = 16
has_ipyparallel = True
parallel_job_count = 6
has_metadata = False

# Code for building the graph

def get_demand_nodes(G):
    return list({x.target_vertex for x in G.es(Tier=1)})

def get_node_tier_from_edge_tier(G):

    # iterate through the nodes and assign each node the minimum tier of the
    # edges leaving it
    for node in G.vs:
        if len(node.out_edges()) > 0:
            node['Tier'] = min([e['Tier'] for e in node.out_edges()])
        else:
            node['Tier'] = 0

def get_firm_df(df=None):
    if df is None:
        df = get_df()

    for col in [
        'Source',
        'Source Name',
        'Source Industry',
        'Source Country',
        'Source Market Cap',
        'Source Employees Global',
        'Source Private',
        'Target',
        'Target Name',
        'Target Industry',
        'Target Country',
        'Target Market Cap',
        'Target Employees Global',
            'Target Private']:
        try:
            df[col]
        except BaseException:
            df[col] = 'nan'

    firm_df = pd.DataFrame(dict(
        ID=df['Source'],
        name=df['Source Name'],
        industry=df['Source Industry'],
        country=df['Source Country'],
        market_cap=df['Source Market Cap'],
        Employees=df['Source Employees Global'],
        private=df['Source Private']))
    firm_df = firm_df.append(pd.DataFrame(dict(
        ID=df['Target'],
        name=df['Target Name'],
        industry=df['Target Industry'],
        country=df['Target Country'],
        market_cap=df['Target Market Cap'],
        Employees=df['Target Employees Global'],
        private=df['Target Private'])))

    for col in ['country', 'industry']:
        firm_df[col] = firm_df[col].astype(str)  # make NaNs into string nan
    firm_df.private = firm_df.private == 'Private Company'
    firm_df = firm_df.drop_duplicates(ignore_index=True)
    firm_df.set_index('ID', inplace=True)
    firm_df['country-industry'] = firm_df['country'] + \
        ' ' + firm_df['industry']
    firm_df.loc[firm_df['Employees'] ==
                '(Invalid Identifier)', 'Employees'] = math.nan

    firm_df['Tier'] = df.groupby('Source').Tier.min()
    firm_df.loc[firm_df.Tier.isna(), 'Tier'] = 0
    firm_df.Tier = firm_df.Tier.astype(int)

    return firm_df


def get_edge_df(df=None):
    if df is None:
        df = get_df()
    return df[['Source', 'Target', 'Tier']]


def get_shortcut_edges(G):
    fdf = get_firm_df()
    fdf = pd.concat([fdf,
                     pd.DataFrame(dict(tier=G.vs['Tier'],
                                       clean_tier=G.vs['clean_tier']),
                                  index=G.vs['name'])],
                    axis=1)
    df = get_df()
    df['Source tier'] = df.Source.map(lambda x: fdf.loc[x]['Tier'])
    df['Target tier'] = df.Target.map(lambda x: fdf.loc[x]['Tier'])

    return df[df['Source tier'] > df['Target tier'] + 1]


def edges():
    df = get_edge_df()
    # not guaranteed to respect order of appearance, but it does in practice
    e, i = pd.factorize(df.Source.append(df.Target, ignore_index=True))

    m = e.size // 2

    ee = np.zeros((m, 2), dtype=int)
    ee[:, 0] = e[:m]
    ee[:, 1] = e[m:]
    e = ee
    return e, i


def directed_igraph(*,
                    giant=True,
                    no_software=True,
                    include_private=True,
                    cut_low_terminal_suppliers=True,
                    reduced_density=False,
                    clean_tiers=True,
                    overwrite_old_tiers=True):

    df = get_df()
    firm_df = get_firm_df(df)
    edge_df = get_edge_df(df)
    G = ig.Graph(directed=True)
    G.add_vertices(firm_df.index)
    G.vs['firm name'] = firm_df['name']
    for attr in ['industry', 'country', 'country-industry', 'Employees']:
        G.vs[attr] = firm_df[attr]
    G.vs['Tier'] = firm_df['Tier']

    G.add_edges(edge_df[['Source', 'Target']].itertuples(index=False))

    G.es['Tier'] = edge_df.Tier.values
    # use min to keep smaller tier value.
    G.simplify(loops=False, combine_edges='min')

    if no_software:
        G = G.induced_subgraph(
            [
                x.index for x in G.vs(
                    lambda x: x['industry'] not in [
                        'Application Software',
                        'IT Consulting and Other Services',
                        'Systems Software',
                        'Advertising',
                        'Movies and Entertainment',
                        'Interactive Home Entertainment'])])

    if cut_low_terminal_suppliers:
        # this is not the same as when we did clean_tiers!
        med_suppliers = get_demand_nodes(G)
        t = [get_terminal_nodes(i, G) for i in med_suppliers]
        # 31 because there is a jump in len(tt) here
        to_delete = [m for m, tt in zip(med_suppliers, t) if len(tt) < 31]
        G.delete_vertices(to_delete)

    if clean_tiers:  # remove nodes not tracing back to tier 0
        med_suppliers = get_demand_nodes(G)
        to_keep = med_suppliers
        curr_tier = to_keep
        G.vs['clean_tier'] = [0] * G.vcount()
        # tier=0
        # while len(curr_tier)>0:
        for tier in range(0, max_tiers):

            # get next tier (as a list with duplicates)
            next_tier = []
            for v in curr_tier:
                next_tier += v.predecessors()

            #  the next current tier will be the new nodes we just found
            curr_tier = set(next_tier).difference(to_keep)
            for v in curr_tier:
                v['clean_tier'] = tier + 1

            # add next tier to the list of nodes to keep
            to_keep = set(to_keep).union(curr_tier)
            # tier += 1

        if overwrite_old_tiers:
            G.vs['Tier'] = G.vs['clean_tier']
            for e in G.es:
                e['Tier'] = e.target_vertex['Tier'] + 1

        G = G.induced_subgraph(to_keep)
    else:
        G.vs['tier_set'] = [{e['Tier'] for e in n.out_edges()}
                            for n in G.vs]
        G.vs['Tier'] = [min(x['tier_set'], default=0) for x in G.vs]

    if giant:
        # this turns out to be a no-op if you do clean-tiers with 10 tiers
        G = G.components(mode='WEAK').giant()

    for v in get_demand_nodes(G):
        v['Tier'] = 0
        try:
            v['tier_set'].add(0)
        except BaseException:
            pass

    if reduced_density:
        print('Reducing density')
        # .2 works. .1 might have no med suppliers
        G = random_thinning_factory(G)(reduced_density)

    med_suppliers = get_demand_nodes(G)
    G.vs['is_demand_node'] = [i in med_suppliers for i in G.vs]
    G.reversed = False

    G[45]

    try:
        for firm in [
            "U.s. Military.",
            "National Aeronautics and Space Administration",
            "United States Department Of Defense",
                "US Navy"]:  # US Navy is not there if you clean tiers
            G.vs(lambda x: x['firm name'] == firm)[
                0]['industry'] = 'U.S. Military'
    except BaseException:
        pass

    return G

# Utility functions


def get_plural(x):
    if x == 'firm':
        return 'firms'
    elif x == 'country':
        return 'countries'
    elif x == 'industry':
        return 'industries'
    elif x == 'country-industry':
        return 'country-industries'
    else:
        raise NotImplementedError


def clean_prefix(prefix):
    if prefix != '' and prefix[-1] != '/':
        prefix += '/'
    return prefix



def reverse(G):
    Tier = dict(Tier=G.es['Tier'])
    edges = [tuple(reversed(e.tuple)) for e in G.es]
    G.delete_edges(None)
    G.add_edges(edges, Tier)
    G.reversed = not G.reversed


def get_terminal_nodes(node, G):
    if isinstance(node, ig.Vertex):
        node = node.index

    reachable_nodes = get_reachable_nodes(node, G)
    reachable_graph = G.induced_subgraph(reachable_nodes)

    sccs = reachable_graph.clusters()

    terminal_components = sccs.cluster_graph().vs(_indegree_eq=0)
    sccs = list(sccs)
    terminal_nodes = [sccs[node.index] for node in terminal_components]
    terminal_nodes = {reachable_graph.vs[node]['name']
                      for node in itertools.chain(*terminal_nodes)}
    return terminal_nodes

# All I need now is a way to efficiently get t_thin (indices of t in G_thin)


def is_reachable(t_thin, i_thin, G_thin):
    return np.array(G_thin.shortest_paths(i_thin, t_thin)) < np.inf


def is_increasing(l):
    if len(l) < 2:
        return True
    for i in range(len(l) - 1):
        if l[i] > l[i + 1]:
            return False
    return True


def is_subgraph(G_thin, G):

    # check nodes
    for i in G_thin.vs['name']:
        if i not in G.vs['name']:
            return False

    # check edges
    for e in G_thin.es:
        s = e.source_vertex['name']
        t = e.target_vertex['name']
        sv = G.vs(name=s)[0]
        tv = G.vs(name=t)[0]
        if tv not in sv.successors():
            return False

    return True


def get_reachable_nodes(node, G):
    if isinstance(node, ig.Vertex):
        node = node.index

    u = G.bfs(node, mode='IN')
    u = u[0][:u[1][-1]]  # remove trailing zeros

    # return the ids of the nodes
    return {G.vs['name'][i] for i in u}

# TODO: make this faster:
#   Change G to the quotient graph (once in the calling function, passed in as an optional argument) G.components().cluster_graph()
#   Then since it is a DAG maybe there is a smarter way to do this
#   Try using is_reachable above instead


def get_u(i_thick, G_thin, med_suppliers_thin=None, direction='IN'):
    if isinstance(i_thick, ig.Vertex):
        i_thick = i_thick.index

    try:
        i_thin = med_suppliers_thin[i_thick] if med_suppliers_thin else G_thin.vs.find(
            name=i_thick).index
    except BaseException:  # the node we want has been deleted
        return set()

    u = G_thin.bfs(i_thin, mode=direction)
    u = u[0][:u[1][-1]]  # remove trailing zeros

    ids = G_thin.vs['name']
    return {ids[i] for i in u}


def impute_industry(G):
    try:
        G['industry_imputed']
    except BaseException:
        G.vs['industry_imputed'] = [x == 'nan' for x in G.vs['industry']]

    industry_dist = np.array([x['industry']
                             for x in G.vs if not x['industry_imputed']])
    imputed_industry = np.random.choice(industry_dist, len(
        G.vs(industry_imputed_eq=True)), replace=True)
    for v, s in zip(G.vs(industry_imputed_eq=True), imputed_industry):
        v['industry'] = s

# Stats to run on disrupted graphs

# i is an integer, the index of the node in G


def some_terminal_suppliers_reachable(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)
    if u is None:
        u = get_u(i, G_thin)

    if u & t:  # set intersection
        return True
    return False


some_terminal_suppliers_reachable.description = 'Some end suppliers reachable'
some_terminal_suppliers_reachable.type = bool


def all_terminal_suppliers_reachable(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)
    if u is None:
        u = get_u(i, G_thin)

    return t.issubset(u)


all_terminal_suppliers_reachable.description = 'All end suppliers reachable'


def percent_terminal_suppliers_reachable(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)
    if u is None:
        u = get_u(i, G_thin)

    return len(set(t) & u) / len(t)


percent_terminal_suppliers_reachable.description = 'Avg. percent end suppliers reachable'
percent_terminal_suppliers_reachable.type = float


def all_terminal_suppliers_exist(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)

    return t.issubset(G_thin.vs['name'])


all_terminal_suppliers_exist.description = 'All end suppliers surviving'

# can't move this up since the callbacks need to be defined
callbacks = [some_terminal_suppliers_reachable,
             percent_terminal_suppliers_reachable]

# Attacks


def random_thinning_factory(G):
    firm_rands = np.random.random(G.vcount())

    uniques = dict()
    perm = dict()
    if has_metadata:
        for failure_scale in ['country', 'industry', 'country-industry']:
            uniques[failure_scale] = list(set(G.vs[failure_scale]))
            perm[failure_scale] = uniques[failure_scale]
            random.shuffle(perm[failure_scale])

    def attack(rho, failure_scale='firm'):
        if failure_scale == 'firm':
            return G.induced_subgraph(
                (firm_rands <= rho).nonzero()[0].tolist())
        else:
            keep_uniques = perm[failure_scale][:round(
                rho * len(uniques[failure_scale]))]
            return G.induced_subgraph(
                G.vs(lambda x: x[failure_scale] in keep_uniques))
    attack.description = 'Random'

    return attack


random_thinning_factory.description = 'Random'


def get_sorted_attr_inds(G, attr):

    sorted_attr_inds = dict()
    sorted_attr_inds['firm'] = sorted(
        range(G.vcount()), key=G.vs[attr].__getitem__)
    for failure_scale in ['country', 'industry', 'country-industry']:
        sorted_attr_inds[failure_scale] = sorted(set(G.vs[failure_scale]), key=lambda x: sum(
            G.vs(lambda v: v[failure_scale] == x)[attr]))
    return sorted_attr_inds


def target_by_attribute(G, attr, protected_countries=[]):

    sorted_attr_inds = get_sorted_attr_inds(G, attr)

    def targeted(r, failure_scale='firm'):
        to_keep = sorted_attr_inds[failure_scale][:int(
            len(sorted_attr_inds[failure_scale]) * r)]
        if failure_scale == 'firm':
            return G.induced_subgraph(
                to_keep +
                list(
                    G.vs(
                        lambda x: x['country'] in protected_countries)))
        else:
            return G.induced_subgraph(G.vs(lambda x: (
                str(x[failure_scale]) in to_keep) or (x['country'] in protected_countries)))

    targeted.description = attr

    return targeted


def get_employee_attack(G, protected_countries=[]):
    try:
        G.vs['Employees_imputed']
    except BaseException:
        G.vs['Employees_imputed'] = [math.isnan(x) for x in G.vs['Employees']]
    size_dist_private = np.array([x['Employees']
                                 for x in G.vs if not x['Employees_imputed']])
    imputed_size = np.random.choice(
        size_dist_private, len(
            G.vs(
                Employees_imputed_eq=True)))
    for v, s in zip(G.vs(Employees_imputed_eq=True), imputed_size):
        v['Employees'] = s
    return target_by_attribute(
        G, 'Employees', protected_countries=protected_countries)


get_employee_attack.description = 'Employees'


def get_degree_attack(G):
    G.vs['degree'] = G.degree(range(G.vcount()))
    return target_by_attribute(G, 'degree')


get_degree_attack.description = 'Degree'


def get_pagerank_attack(G, transpose=True, protected_countries=[]):

    attrname = 'Pagerank of transpose' if transpose else 'Pagerank'
    try:
        G[attrname]
    except BaseException:
        if transpose:
            reverse(G)
            pr = G.pagerank()
            reverse(G)
        else:
            pr = G.pagerank()
        G.vs[attrname] = pr

    return target_by_attribute(
        G, attrname, protected_countries=protected_countries)


get_pagerank_attack.description = 'Pagerank of transpose'


def get_pagerank_attack_no_transpose(G, protected_countries=[]):
    return get_pagerank_attack(
        G,
        transpose=False,
        protected_countries=protected_countries)


get_pagerank_attack_no_transpose.description = 'Pagerank'


def get_null_attack(G):
    def null_attack(r, failure_scale='firm'):
        return G
    return null_attack


get_null_attack.description = 'Null'  # -targeted'

# Failure reachability simulations

n_cpus = len(os.sched_getaffinity(0))
cluster = ipp.Cluster(n = n_cpus - 2)
cluster_is_started = False
def get_dv():
    global cluster_is_started
    if not cluster_is_started:
        cluster.start_cluster_sync()
        cluster_is_started = True
    client = cluster.connect_client_sync()
    client.wait_for_engines()
    dv = client[:]
    dv.use_dill()
    return dv

def single_entity_deletion(G, scale='firm'):
    assert (scale == 'firm')

    med_suppliers = get_demand_nodes(G)
    ts = [get_terminal_nodes(i, G) for i in med_suppliers]

    res = []
    for v in G.vs:
        print(v)
        G_thin = deepcopy(G)
        G_thin.delete_vertices(v)
        med_suppliers_thin = {
            i_thin['name']: i_thin.index for i_thin in G_thin.vs if i_thin['name'] in med_suppliers}
        us = [get_u(i, G_thin, med_suppliers_thin) for i in med_suppliers]
        sample = [
            percent_terminal_suppliers_reachable(
                med_suppliers, G, G_thin, t, u) for i, t, u in zip(
                med_suppliers, ts, us)]
        res.append(np.mean(sample))

    return res

# Plots

def failure_plot(
        avgs,
        plot_title='Supply chain resilience under firm failures',
        save_only=False,
        filename=None):

    rho = avgs.columns[0]
    ax = []
    ax = [
        sns.lineplot(
            x=rho,
            y=col,
            label=col,
            data=avgs,
            errorbar=(
                'pi',
                95)) for col in avgs.columns]
    ax[0].set(xlabel=rho,
              ylabel='Percent of firms',
              title=plot_title)
    plt.legend()

    if save_only:
        plt.savefig(filename)


def attack_compare_plot(
        avgs=None,
        fname='temp.svg',
        save=False,
        failure_scale='firm',
        tiers=max_tiers,
        software_included=False,
        metric=percent_terminal_suppliers_reachable.description,
        rho_scale=np.linspace(.3, 1, 71)):

    rho = "Percent " + get_plural(failure_scale) + " remaining"

    data = avgs[
        (avgs['Failure scale'] == failure_scale) &
        (avgs[rho] <= rho_scale[-1]) &
        (avgs[rho] >= rho_scale[0])]
    ax = sns.lineplot(x=rho,
                      y=metric,
                      data=data,
                      hue='Attack type',
                      errorbar=('pi', 95),
                      legend='full', estimator=np.median)
    plt.plot([rho_scale[0], rho_scale[-1]], [rho_scale[0], rho_scale[-1]],
             color=sns.color_palette()[data['Attack type'].unique().size], label=rho)
    ax.set(title=failure_scale.capitalize() + ' failures')
    plt.legend()

    if save:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)

    return ax


def failure_reachability_single(
        r,
        G,
        med_suppliers=False,
        ts=False,
        failure_scale='firm',
        callbacks=callbacks,
        targeted=False):

    write_test_file()

    if not med_suppliers:
        med_suppliers = get_demand_nodes(G)
    if not ts:
        ts = [set(get_terminal_nodes(i, G)) for i in med_suppliers]
    if not targeted:
        targeted = random_thinning_factory(G)

    G_thin = targeted(r, failure_scale=failure_scale)
    med_suppliers_thin = {
        i_thin['name']: i_thin.index for i_thin in G_thin.vs if i_thin['name'] in med_suppliers}

    res = dict()
    us = [get_u(i, G_thin, med_suppliers_thin) for i in med_suppliers]
    for cb in callbacks:
        sample = [cb(med_suppliers, G, G_thin, t, u)
                  for i, t, u in zip(med_suppliers, ts, us)]
        res[cb.description] = np.mean(sample)
    res['Failure scale'] = failure_scale
    res['Attack type'] = targeted.description
    return res

def write_test_file():
    # save a file whose name contains the time and process id
    import time
    import os
    import sys
    import pickle
    pickle.dump(
            [1,2,3],
            open(
                'G_{}_{}.pkl'.format(
                    time.strftime(
                        "%Y%m%d-%H%M%S"),
                    os.getpid()),
                'wb'))


def failure_reachability_sweep(G,
                               rho=np.linspace(.3,
                                               1,
                                               71),
                               med_suppliers=False,
                               ts=False,
                               failure_scale='firm',
                               callbacks=callbacks,
                               targeted_factory=random_thinning_factory,
                               parallel=False):

    if failure_scale == 'industry':
        G = deepcopy(G)
        impute_industry(G)

    if not med_suppliers:
        med_suppliers = [i.index for i in get_demand_nodes(G)]
    if not ts:
        ts = [set(get_terminal_nodes(i, G)) for i in med_suppliers]

    avgs = []
    if parallel:
        dv = get_dv()
        with dv.sync_imports():
            import sc
        dv['G'] = G
        dv['med_suppliers'] = med_suppliers
        dv['ts'] = ts
        dv['failure_scale'] = failure_scale
        dv['callbacks'] = callbacks
        dv['targeted_factory'] = targeted_factory

        assert(False)

        avgs = dv.map(failure_reachability_single,
                  rho,
                  *list(zip(*[[G,
                               med_suppliers,
                               ts,
                               failure_scale,
                               callbacks,
                               targeted_factory(G)]] * len(rho))))
    else:

        targeted = targeted_factory(G)

        for r in rho:
            print(r)
            avgs.append(
                failure_reachability_single(
                    r,
                    G,
                    med_suppliers,
                    ts,
                    failure_scale=failure_scale,
                    callbacks=callbacks,
                    targeted=targeted))

    avgs = [pd.DataFrame(a, index=[0]) for a in avgs]
    avgs = pd.concat(avgs, ignore_index=True)
    rho_name = "Percent " + get_plural(failure_scale) + " remaining"
    avgs[rho_name] = rho
    cols = list(avgs.columns)
    avgs = avgs[cols[-1:] + cols[:-1]]

    return avgs


def failure_reachability(G,
                         rho=np.linspace(.3, 1, 71),
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

    if parallel == 'auto' or parallel:
        parallel = 'repeat' if repeats > 1 else 'rho'

    if med_suppliers is None:
        med_suppliers = [i.index for i in get_demand_nodes(G)]

    if use_cached_t:
        t = []
        with open('.cached_t', 'rb') as f:
            t = pickle.load(f)
    else:
        t = [get_terminal_nodes(i, G) for i in med_suppliers]
        with open('.cached_t', 'wb') as f:
            pickle.dump(t, f)

    args = [[G, rho, med_suppliers, t, failure_scale, callbacks, targeted_factory]
            ] * repeats  # Beware here that the copy here is very shallow

    if parallel == 'repeat':
        print('doing parallel map now')
        dv = get_dv()
        with dv.sync_imports():
            import sc
        dv['G'] = G
        dv['med_suppliers'] = med_suppliers
        dv['t'] = t
        dv['failure_scale'] = failure_scale
        dv['callbacks'] = callbacks
        dv['targeted_factory'] = targeted_factory

        #def wrapper(x):
            #return sc.failure_reachability_sweep(G, med_suppliers, t, rho, failure_scale, callbacks, targeted_factory)

        #avgs = dv.map(wrapper, range(repeats))
        avgs = dv.map(lambda x: sc.failure_reachability(G, med_suppliers, t, rho, failure_scale, callbacks, targeted_factory), range(repeats))
#
#        avgs = dv.map(failure_reachability_sweep,
#                    *list(zip(*args)))
    elif parallel == 'rho':
        avgs = [failure_reachability_sweep(*args[0], parallel=True)]
    else:
        avgs = [failure_reachability_sweep(*args[0]) for _ in range(repeats)]
    avgs = pd.concat(avgs, ignore_index=True)

    if plot:
        plot_title = targeted_factory.description.capitalize() + ' '\
            + failure_scale + ' failures'\
            + ((' excluding software firms' if G_has_no_software_flag else ' including software firms') if G_has_no_software_flag is not None else '')
        prefix = clean_prefix(prefix)
        fname = failure_scale\
            + '_' + targeted_factory.description.replace(' ', '_').lower()\
            + '_range_' + str(rho[0]) + '_' + str(rho[-1])\
            + '_repeats_' + str(repeats)\
            + (('software_excluded' if G_has_no_software_flag else 'software_included') if G_has_no_software_flag is not None else '')
        if save_only:
            os.makedirs(
                os.path.dirname(
                    prefix +
                    'dat/' +
                    fname),
                exist_ok=True)
            with open(prefix + 'dat/' + fname + '.pickle', mode='wb') as f:
                pickle.dump(avgs, f)
        os.makedirs(os.path.dirname(prefix + 'im/' + fname), exist_ok=True)
        failure_plot(avgs[avgs.columns[:-2]],
                     plot_title=plot_title,
                     save_only=save_only,
                     filename=prefix + 'im/' + fname + '.svg')

    return avgs

# Tiers stuff


def reduce_tiers(G, tiers):
    # This can delete some edges even if tier=max_tier, since there can be
    # edges of tier max_tier+1
    G.delete_edges(G.es(Tier_ge=tiers + 1))
    G.delete_vertices(G.vs(Tier_ge=tiers + 1))
    G.vs['name'] = list(range(G.vcount()))
    for attr in [
        'Pagerank',
        'Pagerank of transpose',
        'Employees_imputed',
            'Industry_imputed']:
        try:
            del G.vs[attr]
        except BaseException:
            pass


def compare_tiers_plot(res,
                       rho=np.linspace(.3,
                                       1,
                                       71),
                       failure_scale='firm',
                       attack=random_thinning_factory,
                       save=True,
                       prefix=''):
    rho = "Percent " + get_plural(failure_scale) + " remaining"
    ax = sns.lineplot(
        x=rho,
        y=percent_terminal_suppliers_reachable.description,
        data=res,
        hue='Tier count',
        errorbar=('pi', 95),
        legend='full')
    ax.set(title=attack.description.capitalize() + ' failures')
    if save:
        fname = failure_scale\
            + '_' + attack.description.replace(' ', '_').lower()\
            + '_range_' + str(rho[0]) + '_' + str(rho[-1])\
            + '_tiers_' + str(res['Tier count'].min()) + '_' + str(res['Tier count'].max())
        os.makedirs(prefix + 'im/' + os.path.dirname(fname), exist_ok=True)
        plt.savefig(prefix + 'im/' + fname + '.svg')


def compare_tiers(G,
                  rho=np.linspace(.3, 1, 71),
                  repeats=24,
                  plot=True,
                  save=True,
                  attack=random_thinning_factory,
                  failure_scale='firm',
                  tier_range=range(1, max_tiers + 1),
                  prefix='',
                  parallel='auto'):
    G = deepcopy(G)
    res = pd.DataFrame()
    for tiers in reversed(tier_range):
        print(tiers)
        reduce_tiers(G, tiers)
        res_tier = failure_reachability(
            G,
            rho=rho,
            plot=False,
            callbacks=(percent_terminal_suppliers_reachable,),
            repeats=repeats,
            targeted_factory=attack,
            failure_scale=failure_scale,
            parallel=parallel)
        res_tier['Tier count'] = tiers
        res = res.append(res_tier, ignore_index=True)
        with open(prefix + 'temp.pickle', mode='wb') as f:  # save in case there is a crash
            pickle.dump(res, f)

    fname = 'compare_tiers/' + failure_scale + '/' + \
        attack.description.replace(' ', '_').lower()
    os.makedirs(prefix + 'dat/' + os.path.dirname(fname), exist_ok=True)
    res.to_hdf(prefix + 'dat/' + fname + '.h5', key='res')

    if plot:
        compare_tiers_plot(res, rho, failure_scale, attack, save, prefix)
    return res


def required_tiers(res, attack, scale):
    # Assume res is already filtered to a specific attack and scale

    res = res[(res['Failure scale'] == scale) & (
        res['Attack type'] == attack.description)]
    rho = "Percent " + get_plural(scale) + " remaining"

    means = []
    for tier_count in range(1, max_tiers + 1):
        means.append(res[res['Tier count'] == tier_count].groupby(
            rho)['Avg. percent end suppliers reachable'].mean())

    maxes = np.zeros(max_tiers)
    for tier_count in range(1, max_tiers + 1):
        maxes[tier_count -
              1] = np.max(np.abs(means[tier_count - 1] - means[-1]))

    tol = .05
    return np.nonzero(maxes < tol)[0][0] + 1

