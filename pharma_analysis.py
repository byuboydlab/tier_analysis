import sc
import pandas as pd
import igraph as ig
import numpy as np

breakdown_threshold = 0.95
thinning_ratio = 0.002
repeats_per_node = 20
parallel_job_count = 68

edge_df = pd.read_csv('dat/pharma_supply_chain.csv')
edge_df.drop('Unnamed: 0',axis=1,inplace=True)
edge_df.rename(columns={'source':"Source", 'target':"Target", 'tier':"Tier"},inplace=True)

G = sc.igraph_simple(edge_df)
sc.get_node_tier_from_edge_tier(G)

def get_node_breakdown_threshold(node, G, breakdown_threshold, thinning_ratio):

    # get terminal nodes for node
    terminal_nodes = sc.get_terminal_nodes(node, G)

    # repeatedly delete thinning_ratio percent of nodes from G until there is no path from node to at least breakdown_threshold percent of the farther upstream nodes
    G_thin = G.copy()
    reachable_node_count = len(terminal_nodes)
    while reachable_node_count >= breakdown_threshold * len(terminal_nodes):

        # delete thinning_ratio percent of nodes from G_thin
        to_delete = G_thin.vs(np.random.randint(0,len(G_thin.vs),int(thinning_ratio*len(G_thin.vs))))
        G_thin.delete_vertices(to_delete)

        # reachable node count
        # find node in G_thin that corresponds to node in G
        try:
            node_thin = G_thin.vs.select(name=node['name'])[0]
        except:
            break # node was deleted

        reachable_node_count = len(sc.get_reachable_nodes(node_thin.index, G_thin) & set(terminal_nodes))

    # store number of nodes deleted
    node['Deleted count'] = len(G.vs) - len(G_thin.vs)

    return len(G.vs) - len(G_thin.vs)

if __name__ == '__main__':
    itercount = 0
    thresholds = pd.DataFrame(np.zeros((len(G.vs.select(Tier=0)),repeats_per_node)),index=G.vs.select(Tier=0)['name'],columns=list(range(repeats_per_node)))

    parallel=True
    if parallel:
        import ipyparallel
        with ipyparallel.Cluster(n=68) as rc:
            #rc = cluster.start_and_connect_sync()
            rc.wait_for_engines(68)
            dv = rc[:]
            #dv = ipyparallel.Client()[:] # This should be global (or a singleton) to avoid an error with too many files open https://github.com/ipython/ipython/issues/6039
            dv.block=False
            dv.use_dill()
            with dv.sync_imports():
                from pharma_analysis import get_node_breakdown_threshold
    
            def repeat_breakdown_test(node,G, breakdown_threshold, thinning_ratio, repeats_per_node):
                return [get_node_breakdown_threshold(G.vs[node], G, breakdown_threshold, thinning_ratio) for _ in range(repeats_per_node)]
            
            res = dv.map_sync(repeat_breakdown_test, 
                    [v.index for v in G.vs.select(Tier=0)], 
                    [G for _ in G.vs.select(Tier=0)], 
                    [breakdown_threshold for _ in G.vs.select(Tier=0)], 
                    [thinning_ratio for _ in G.vs.select(Tier=0)],
                    [repeats_per_node for _ in G.vs.select(Tier=0)])
    else:
        for node in G.vs.select(Tier=0):
            # print progress bar
            print('Progress: {0:.2f}%'.format(100*itercount/len(G.vs.select(Tier=0))),end='\r')
            for i in range(repeats_per_node):
                thresholds.loc[node['name'], i] = get_node_breakdown_threshold(node, G, breakdown_threshold, thinning_ratio)
            itercount += 1
