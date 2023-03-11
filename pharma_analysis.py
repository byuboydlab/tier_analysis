import sc
import pandas as pd
import igraph as ig
import numpy as np

breakdown_threshold = 0.99
thinning_ratio = 0.02
repeats_per_node = 2
parallel = True
parallel_job_count = 5
parallel_node_limit = 10  # for testing. set to None to run all nodes

edge_df = pd.read_csv('dat/pharma_supply_chain.csv')
edge_df.drop('Unnamed: 0', axis=1, inplace=True)
edge_df.rename(
    columns={
        'source': "Source",
        'target': "Target",
        'tier': "Tier"},
    inplace=True)

G = sc.igraph_simple(edge_df)
sc.get_node_tier_from_edge_tier(G)


def get_node_breakdown_threshold(node, G, breakdown_threshold, thinning_ratio):

    # get terminal nodes for node
    terminal_nodes = sc.get_terminal_nodes(node, G)

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
        except BaseException:
            break  # node was deleted

        reachable_node_count = len(
            sc.get_reachable_nodes(
                node_thin.index,
                G_thin) & set(terminal_nodes))

    # store number of nodes deleted
    node['Deleted count'] = len(G.vs) - len(G_thin.vs)

    return len(G.vs) - len(G_thin.vs)


if __name__ == '__main__':
    itercount = 0

    nodes = G.vs.select(Tier=0)
    if parallel_node_limit:
        nodes = nodes[:parallel_node_limit]

    thresholds = pd.DataFrame(
        np.zeros(
            (len(nodes), repeats_per_node)), index=nodes['name'], columns=list(
            range(repeats_per_node)))

    if parallel:
        import ipyparallel
        with ipyparallel.Cluster(n=parallel_job_count) as rc:
            rc.wait_for_engines(parallel_job_count)
            lv = rc.load_balanced_view()
            lv.block = False
            rc[:].use_dill()
            with rc[:].sync_imports():
                from pharma_analysis import get_node_breakdown_threshold

            def repeat_breakdown_test(
                    node,
                    G,
                    breakdown_threshold,
                    thinning_ratio,
                    repeats_per_node):
                return [
                    get_node_breakdown_threshold(
                        G.vs[node],
                        G,
                        breakdown_threshold,
                        thinning_ratio) for _ in range(repeats_per_node)]

            res = lv.map(repeat_breakdown_test,
                         [v.index for v in nodes],
                         [G for _ in nodes],
                         [breakdown_threshold for _ in nodes],
                         [thinning_ratio for _ in nodes],
                         [repeats_per_node for _ in nodes])
            res.wait_interactive()
            res = res.get()
            # change res to df indexed by node
            for i, node in enumerate(nodes):
                thresholds.loc[node['name'], :] = res[i]
            # save thresholds to excel file with breakdown threshold and
            # thinning ratio in filename
            fname = 'dat/pharma_thresholds_{0:.2f}_{1:.3f}.xlsx'.format(
                breakdown_threshold, thinning_ratio)
            # add date and time to filename
            import datetime
            fname = fname[:-5] + '_' + \
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + fname[-5:]
            thresholds.to_excel(fname)
    else:
        for node in G.vs.select(Tier=0):
            # print progress bar
            print('Progress: {0:.2f}%'.format(
                100 * itercount / len(G.vs.select(Tier=0))), end='\r')
            for i in range(repeats_per_node):
                thresholds.loc[node['name'], i] = get_node_breakdown_threshold(
                    node, G, breakdown_threshold, thinning_ratio)
            itercount += 1
