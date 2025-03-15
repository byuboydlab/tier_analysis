import os
import sys
import math
import datetime
import tomllib
import itertools
import random
import tqdm
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
import dask.distributed as dist
from copy import deepcopy


with open(sys.argv[2], 'rb') as config_file:
    config = tomllib.load(config_file)


def get_df(extra_tiers=False):
    global file_name

    files = list(os.scandir())
    files = [x for x in files if x.is_file() and x.name == sys.argv[1]]
    if len(files) == 0:
        raise Exception('No files match the data file name given!')
    else:
        file_name = files[0]

    df = pd.read_excel(file_name, sheet_name="Sheet1", engine='openpyxl')
    df = df.drop_duplicates(ignore_index=True)

    try:
        df = df[df['Relationship Type'] == 'Supplier']
        df.reset_index()
    except BaseException:
        pass

    # resolve NaNs for better typing
    for col in [
        'Source Country',
        'Target Country',
        'Source Name',
        'Target Name',
        'Source Industry',
        'Target Industry',
        'Source Private',
            'Target Private']:
        try:  # in case these columns are not there
            df[col] = df[col].astype(str)
        except BaseException:
            pass
    for col in [
        'Source Market Cap',
        'Target Market Cap',
        'Source Revenue',
        'Target Revenue',
        'Source Employees Global',
            'Target Employees Global']:
        try:  # in case these columns are not there
            df.loc[df[col] == '(Invalid Identifier)', col] = math.nan
            df[col] = df[col].astype(float)
        except BaseException:
            pass

    return df


def get_demand_nodes(G):
    return list({x.target_vertex for x in G.es(Tier=1)})


def igraph_simple(edge_df):

    firm_list = pd.concat((edge_df['Source'], edge_df['Target'])).unique()
    G = ig.Graph(directed=True)
    G.add_vertices(firm_list)
    G.add_edges(edge_df[['Source', 'Target']].itertuples(index=False))
    G.es['Tier'] = edge_df.Tier.values
    # use min to keep smaller tier value.
    G.simplify(loops=False, combine_edges='min')
    G.reversed = False

    return G


def get_node_tier_from_edge_tier(G):

    # iterate through the nodes and assign each node the minimum tier of the
    # edges leaving it
    for node in G.vs:
        if len(node.out_edges()) > 0:
            node['Tier'] = min([e['Tier'] for e in node.out_edges()])
        else:
            node['Tier'] = 0


def get_reachable_nodes(node, G):
    if isinstance(node, ig.Vertex):
        node = node.index

    u = G.bfs(node, mode='IN')
    u = u[0][:u[1][-1]]  # remove trailing zeros

    # return the ids of the nodes
    return {G.vs['name'][i] for i in u}


def get_terminal_nodes(node, G):
    if isinstance(node, ig.Vertex):
        node = node.index

    reachable_nodes = get_reachable_nodes(node, G)
    reachable_graph = G.induced_subgraph(reachable_nodes)

    sccs = reachable_graph.connected_components()

    terminal_components = sccs.cluster_graph().vs(_indegree_eq=0)
    sccs = list(sccs)
    terminal_nodes = [sccs[node.index] for node in terminal_components]
    terminal_nodes = {reachable_graph.vs[node]['name']
                      for node in itertools.chain(*terminal_nodes)}
    return terminal_nodes


def get_upstream(i_thick, G_thick, G_thin, demand_nodes_thin=None, direction='IN'):
    if isinstance(i_thick, ig.Vertex):
        i_thick = i_thick.index

    try:
        # i_thin = demand_nodes_thin[i_thick] if demand_nodes_thin else G_thin.vs.find(name=i_thick).index
        if demand_nodes_thin:
            i_thin = demand_nodes_thin[i_thick]
        else:
            i_thin = G_thin.vs.find(name=G_thick.vs[i_thick]['name']).index
    except ValueError:  # the node we want has been deleted
        return set()

    upstream = G_thin.bfs(i_thin, mode=direction)
    upstream = upstream[0][:upstream[1][-1]]  # remove trailing zeros

    ids = G_thin.vs['name']
    upstream_nodes = {ids[i] for i in upstream}

    # Add the original node to the results
    upstream_nodes = upstream_nodes.union({ids[i_thin]})

    return upstream_nodes


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


def some_terminal_suppliers_reachable(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)
    if u is None:
        u = get_upstream(i, G, G_thin)

    if len(u.intersection(t)) > 0:  # set intersection
        return True
    return False


some_terminal_suppliers_reachable.description = 'Some end suppliers reachable'
some_terminal_suppliers_reachable.type = bool


def percent_terminal_suppliers_reachable(i, G, G_thin, t=None, u=None):
    if t is None:
        t = get_terminal_nodes(i, G)
    if u is None:
        u = get_upstream(i, G, G_thin)

    return len(t.intersection(u)) / len(t)


percent_terminal_suppliers_reachable.description = 'Avg. percent end suppliers reachable'
percent_terminal_suppliers_reachable.type = float


callbacks = [some_terminal_suppliers_reachable,
             percent_terminal_suppliers_reachable]


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


def reverse(G):
    Tier = dict(Tier=G.es['Tier'])
    edges = [tuple(reversed(e.tuple)) for e in G.es]
    G.delete_edges(None)
    G.add_edges(edges, Tier)
    G.reversed = not G.reversed


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


def random_thinning_factory(G):
    firm_rands = np.random.random(G.vcount())

    uniques = dict()
    perm = dict()
    if config['general']['has_metadata']:
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


def failure_reachability_single(
        r,
        G,
        demand_nodes=False,
        ts=False,
        failure_scale='firm',
        callbacks=callbacks,
        targeted=False):

    if not demand_nodes:
        demand_nodes = get_demand_nodes(G)
    if not ts:
        ts = [set(get_terminal_nodes(i, G)) for i in demand_nodes]
    if not targeted:
        targeted = random_thinning_factory(G)

    G_thin = targeted(r, failure_scale=failure_scale)
    demand_nodes_thin = {
        i_thin['name']: i_thin.index for i_thin in G_thin.vs if i_thin['name'] in demand_nodes}

    res = dict()
    us = [get_upstream(i, G, G_thin, demand_nodes_thin) for i in demand_nodes]
    for cb in callbacks:
        sample = [cb(demand_nodes, G, G_thin, t, u)
                  for i, t, u in zip(demand_nodes, ts, us)]
        res[cb.description] = np.mean(sample)
    res['Failure scale'] = failure_scale
    res['Attack type'] = targeted.description
    return res


def failure_reachability_sweep(G,
                               rho=np.linspace(.3,
                                               1,
                                               71),
                               demand_nodes=False,
                               ts=False,
                               failure_scale='firm',
                               callbacks=callbacks,
                               targeted_factory=random_thinning_factory,
                               parallel=False):
    # global failure_reachability_sweep

    if failure_scale == 'industry':
        G = deepcopy(G)
        impute_industry(G)

    if not demand_nodes:
        demand_nodes = [i.index for i in get_demand_nodes(G)]
    if ts == False:
        ts = [set(get_terminal_nodes(i, G)) for i in demand_nodes]

    avgs = []
    if parallel == 'rho' or parallel == 'all':
        client = dist.get_client()
        avgs = client.map(failure_reachability_single,
                  rho,
                  *list(zip(*[[G,
                               demand_nodes,
                               ts,
                               failure_scale,
                               callbacks,
                               targeted_factory(G)]] * len(rho))))
        avgs = client.gather(avgs)
    else:

        targeted = targeted_factory(G)

        for r in tqdm.tqdm(rho, desc='Reachability sweep', leave=False):
            avgs.append(
                failure_reachability_single(
                    r,
                    G,
                    demand_nodes,
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
                         targeted_factory=random_thinning_factory,
                         parallel='auto',
                         callbacks=callbacks,
                         G_has_no_software_flag=None,
                         prefix='',
                         demand_nodes=None):

    global start_time

    # Check that G is an igraph
    if not isinstance(G, ig.Graph):
        raise ValueError('G must be an igraph.Graph')


    if parallel == 'auto':
        parallel = 'repeat' if repeats > 1 else 'rho'

    if demand_nodes is None:
        demand_nodes = [i.index for i in get_demand_nodes(G)]

    t = [get_terminal_nodes(i, G) for i in demand_nodes]

    args = [[G, rho, demand_nodes, t, failure_scale, callbacks, targeted_factory]
            ] * repeats  # Beware here that the copy here is very shallow


    if parallel == 'repeat' or parallel == 'all':
        print('Doing parallel map now.')
        client = dist.get_client()

        def wrapper_function(x):
            return failure_reachability_sweep(G=G,
                    rho=rho,
                    demand_nodes = demand_nodes,
                    ts = t,
                    failure_scale = failure_scale,
                    callbacks = callbacks,
                    targeted_factory = targeted_factory,
                    parallel = parallel)

        avgs = client.map(wrapper_function, range(repeats))
        avgs = client.gather(avgs)
    elif parallel == 'rho':
        avgs = [failure_reachability_sweep(*args[0], parallel='rho')]
    else:
        avgs = [failure_reachability_sweep(*args[0]) for _ in tqdm.tqdm(range(repeats), desc='Reachability tier', leave=False)]
    avgs = pd.concat(avgs, ignore_index=True)

    if plot:
        plot_title = targeted_factory.description.capitalize() + ' '\
            + failure_scale + ' failures'\
            + ((' excluding software firms' if G_has_no_software_flag else ' including software firms') if G_has_no_software_flag is not None else '')
        fname = failure_scale\
            + '_' + targeted_factory.description.replace(' ', '_').lower()\
            + '_range_' + str(rho[0]) + '_' + str(rho[-1])\
            + '_repeats_' + str(repeats)\
            + (('software_excluded' if G_has_no_software_flag else 'software_included') if G_has_no_software_flag is not None else '')\
            + sys.argv[1].replace('.xlsx', '') + '_' + start_time
        failure_plot(avgs[avgs.columns[:-2]],
                     plot_title=plot_title,
                     save_only=save_only,
                     filename= results_dir + fname + '.svg')

    return avgs


def reduce_tiers(G, tiers):
    # This can delete some edges even if tier=max_tier, since there can be
    # edges of tier max_tier+1
    G = deepcopy(G)
    G.delete_edges(G.es(Tier_ge=tiers + 1))
    G.delete_vertices(G.vs(Tier_ge=tiers + 1))
    for attr in [
        'Pagerank',
        'Pagerank of transpose',
        'Employees_imputed',
            'Industry_imputed']:
        try:
            del G.vs[attr]
        except BaseException:
            pass
    return G


def compare_tiers_plot(res,
                       rho=np.linspace(.3, 1, 71),
                       failure_scale='firm',
                       attack=random_thinning_factory,
                       save=True):

    global start_time

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
            + '_tiers_' + str(res['Tier count'].min()) + '_' + str(res['Tier count'].max())\
            + '_' + sys.argv[1].replace('.xlsx', '') + '_' + start_time
        plt.savefig(results_dir + fname + '.svg')


def compare_tiers(G,
                  rho=np.linspace(.3, 1, 71),
                  repeats=24,
                  plot=True,
                  save=True,
                  attack=random_thinning_factory,
                  failure_scale='firm',
                  tier_range=range(1, config['general']['max_tiers'] + 1),
                  parallel='auto'):
    """
    This function is used to compare the effect of different tier counts on the
    reachability of terminal suppliers.

    Returns:
    res: a dataframe with the results of the reachability for each tier
    """

    global start_time

    G = deepcopy(G) # We don't want to modify the original graph
    res = pd.DataFrame() # Final results
    for tiers in tqdm.tqdm(reversed(tier_range), total=len(tier_range), desc = 'Compare tiers'): # iterate over the number of tiers included
        G = reduce_tiers(G, tiers) # Reduce the graph to the desired number of tiers

        # Call failure_reachability with the reduced graph
        res_tier = failure_reachability(
            G,
            rho=rho,
            plot=False,
            callbacks=(percent_terminal_suppliers_reachable,),
            repeats=repeats,
            targeted_factory=attack,
            failure_scale=failure_scale,
            parallel=parallel)

        # add results to res
        res_tier['Tier count'] = tiers
        res = pd.concat([res, res_tier], ignore_index=True)

    # Save the results
    fname = 'compare_tiers_' + failure_scale + '_' + \
        attack.description.replace(' ', '_').lower()\
        + '_' + sys.argv[1].replace('.xlsx', '') + '_' + start_time
    res.to_excel(results_dir + fname + '.xlsx')

    if plot:
        compare_tiers_plot(res, rho, failure_scale, attack, save)

    return res


def uniform_distance(v1, v2):
    """ Returns the maximum absolute difference between two vectors. """
    return np.max(np.abs(v1 - v2))

def between_tier_distances(res, rho = "Percent firms remaining", attack=random_thinning_factory, failure_scale='firm'):
    """
    Computes the uniform distance between the mean of each tier and the mean of the final tier.

    Parameters:
    - res: DataFrame containing the results.
    - rho: The column name for 'Percent <scale> remaining'.

    Returns:
    - DataFrame with two columns: 'Tier count' and 'Distance'.
    """
    global start_time

    means = {tier_count: res[res['Tier count'] == tier_count].groupby(rho)['Avg. percent end suppliers reachable'].mean()
             for tier_count in res['Tier count'].unique()}

    # Find the distance to the last tier for each tier
    distances = {tier_count: uniform_distance(means[tier_count], means[max(means.keys())])
                 for tier_count in means.keys()}

    # Convert distances dictionary to a DataFrame
    distances_df = pd.DataFrame(list(distances.items()), columns=['Tier count', 'Distance'])

    fname = 'between_tier_distances_' + failure_scale + '_' + \
        attack.description.replace(' ', '_').lower() + '_' + sys.argv[1].replace('.xlsx', '') + '_' + start_time + '.xlsx'
    distances_df.to_excel(results_dir + fname)

    return distances_df


def get_node_breakdown_threshold(node, G, breakdown_threshold=config['breakdown_thresholds']['breakdown_threshold'], thinning_ratio=config['breakdown_thresholds']['thinning_ratio']):

    # if node is int, convert to vertex
    if isinstance(node, int):
        node = G.vs[node]

    # get terminal nodes for node
    terminal_nodes = get_terminal_nodes(node, G)

    # repeatedly delete thinning_ratio percent of nodes from G until there is
    # no path from node to at least breakdown_threshold percent of the farther
    # upstream nodes
    G_thin = G.copy()
    reachable_node_count = len(terminal_nodes)
    while reachable_node_count >= breakdown_threshold * len(terminal_nodes):

        # delete thinning_ratio percent of nodes from G_thin
        to_delete = G_thin.vs(np.random.randint(
            0, G_thin.vcount(), int(thinning_ratio * G_thin.vcount())))
        G_thin.delete_vertices(to_delete)

        # reachable node count
        # find node in G_thin that corresponds to node in G
        try:
            node_thin = G_thin.vs.select(name=node['name'])[0]
        except (ValueError, IndexError):
            break  # node was deleted

        reachable_node_count = len(
            get_reachable_nodes(
                node_thin.index,
                G_thin).union(terminal_nodes))

    # store number of nodes deleted
    node['Deleted count'] = len(G.vs) - len(G_thin.vs)

    return len(G.vs) - len(G_thin.vs)


if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    if len(sys.argv) != 4:
        raise IndexError(f'3 arguments were expected, but {len(sys.argv) - 1} were given.')
    
    results_dir = sys.argv[3]

    if results_dir[-1] != '/' and os.name == 'posix':
        results_dir += '/'
    elif results_dir[-1] != '\\' and results_dir[-1] != '/' and os.name == 'nt':
        results_dir += '/'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if config['parallel']['tiers_parallel_mode'] or config['parallel']['thresholds_parallel']:
        if os.name == 'posix':
            n_cpus = len(os.sched_getaffinity(0))
        elif os.name == 'nt':
            n_cpus = os.cpu_count()
        n = n_cpus - 2
        client = dist.Client(n_workers = n)
    else:
        client = None

    df = get_df()
    G = igraph_simple(df)
    get_node_tier_from_edge_tier(G)

    if config['operations']['compare_tiers']:
        if config['general']['attack_type'] == 'Random':
            factory = random_thinning_factory
        elif config['general']['attack_type'] == 'Employee':
            factory = get_employee_attack
        elif config['general']['attack_type'] == 'Degree':
            factory = get_degree_attack
        elif config['general']['attack_type'] == 'Pagerank':
            factory = get_pagerank_attack_no_transpose
        elif config['general']['attack_type'] == 'Pagerank transpose':
            factory = get_pagerank_attack
        else:
            raise ValueError("Valid values of attack_type are 'Random', 'Employee', 'Degree', 'Pagerank', and 'Pagerank transpose'")

        res = compare_tiers(G, parallel = config['parallel']['tiers_parallel_mode'], attack = factory)
        dists = between_tier_distances(res)
        print(dists)

    if config['operations']['get_thresholds']:

        itercount = 0

        # get nodes with at least reachable_node_threshold of reachable nodes
        nodes = G.vs.select(Tier=0)
        reachability_counts = pd.DataFrame(data=np.zeros(len(nodes)), index=nodes['name'], columns=['counts'])
        
        for node in nodes:
            reachability_counts.at[node['name'], 'counts'] = len(get_reachable_nodes(node, G))


        reachability_counts = reachability_counts[reachability_counts['counts'] >= config['breakdown_thresholds']['reachable_node_threshold']] # cutoff to exclude nodes with few reachable nodes
        nodes = nodes.select(name_in=reachability_counts.index)

        thresholds = pd.DataFrame(
            np.zeros(
                (len(nodes), config['breakdown_thresholds']['repeats_per_node'])), index=nodes['name'], columns=list(
                range(config['breakdown_thresholds']['repeats_per_node'])))

        if config['parallel']['thresholds_parallel']:
            def repeat_breakdown_test(node, repeat_idx):
                res = get_node_breakdown_threshold(G.vs[node], G, config['breakdown_thresholds']['breakdown_threshold'], config['breakdown_thresholds']['thinning_ratio'])
                return res

            pairs = [(v.index, i)
                    for v,i in itertools.product(nodes, range(config['breakdown_thresholds']['repeats_per_node']))]

            res = client.map(repeat_breakdown_test,
                            *zip(*pairs))

            res = client.gather(res)

            for i, (v_idx, i_idx) in enumerate(pairs):
                thresholds.loc[G.vs[v_idx]['name'], i_idx] = res[i]

        else:
            for node in nodes:
                # print progress bar
                print('Progress: {0:.2f}%'.format(
                    100 * itercount / len(nodes)), end='\r')
                for i in range(config['breakdown_thresholds']['repeats_per_node']):
                    thresholds.loc[node['name'], i] = get_node_breakdown_threshold(
                        node, G, config['breakdown_thresholds']['breakdown_threshold'], config['breakdown_thresholds']['thinning_ratio'])
                itercount += 1

        fname = 'breakdown_thresholds_{0:.2f}_{1:.3f}'.format(config['breakdown_thresholds']['breakdown_threshold'], config['breakdown_thresholds']['thinning_ratio'])
        fname = fname + '_' + sys.argv[1].replace('.xlsx', '') + '_' + start_time + '.xlsx'

        thresholds.to_excel(results_dir + fname)

        print('\n')

    print('Complete')
