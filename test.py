from pharma_analysis import *
import igraph as ig
import sc

def make_depth_1_tree():
    G = ig.Graph(directed=True)
    G.add_vertex()
    vcount = 1000
    G.add_vertices(vcount)
    G.add_edges([(i,0) for i in range(1,vcount+1)])
    G.vs['name'] = list(range(G.vcount())) # helps when passing to subgraphs
    G.vs['Tier'] = 1
    G.vs[0]['Tier'] = 0

    return G

def make_depth_2_tree():
    G = make_depth_1_tree()
    degree = 10

    node_counter = G.vcount()
    for node in G.vs.select(Tier=1):
        G.add_vertices(degree, attributes = {'name':list(range(node_counter,node_counter+degree))})
        G.add_edges([(i,node.index) for i in range(node_counter,node_counter+degree)])
        node_counter += degree

    # put all the new nodes in tier 2
    G.vs.select(Tier=None)['Tier'] = 2

    return G


# test to make sure get_node_breakdown_threshold works when the graph is a tree
def test_get_node_breakdown_threshold():
    test_get_node_breakdown_threshold_tree_1()
    test_get_node_breakdown_threshold_tree_2()
    test_get_node_breakdown_threshold_pole_and_twig()

def make_pole_and_twig():
    # make a directed line graph with 1000 nodes
    G = ig.Graph(directed=True)
    G.add_vertices(1000)
    G.add_edges([(i+1,i) for i in range(999)])
    G.vs['name'] = list(range(G.vcount())) # helps when passing to subgraphs
    # assign each edge to a tier based on distance from the root
    G.es['Tier'] = [i for i in range(999)]
    # add an additional node to give the leaf indegree 2
    G.add_vertex()
    G.add_edge(1000,0)
    G.vs[1000]['name'] = 1000
    return G

def test_get_node_breakdown_threshold_pole_and_twig():
    G = make_pole_and_twig()
    threshold = get_node_breakdown_threshold(G.vs.select(name=0)[0], G, breakdown_threshold=.95, thinning_ratio = 1/1000)
    assert(threshold == 1)

def test_get_node_breakdown_threshold_tree_1():
    # make a graph that has one node and 1000 neighbors
    G = make_depth_1_tree()
    node = G.vs.select(name=0)[0]

    # make sure the breakdown threshold is within 1 of 50 for a .95 threshold
    # need standard deviation of binomial distribution
    thinning_ratio = 5/1000
    sd = np.sqrt(G.vcount()*thinning_ratio*(1-thinning_ratio))
    assert abs(get_node_breakdown_threshold(node, G, breakdown_threshold=.95, thinning_ratio = thinning_ratio) - 50) <= 2*sd

def test_get_node_breakdown_threshold_tree_2():
    # make a graph that has one node and 1000 neighbors
    G = make_depth_2_tree()
    node = G.vs.select(name=0)[0]

    # make sure the breakdown threshold is within 1 of 50 for a .95 threshold
    # need standard deviation of binomial distribution
    thinning_ratio = 5/1000
    sd = np.sqrt(G.vcount()*thinning_ratio*(1-thinning_ratio))
    # solve for the number of nodes that need to be deleted
    #assert abs(get_node_breakdown_threshold(node, G, breakdown_threshold=.95, thinning_ratio = thinning_ratio) - 50) <= 2*sd
    
def test_get_terminal_nodes():
    test_get_terminal_nodes_1_tree()
    test_get_terminal_nodes_pole_and_twig()

def test_get_terminal_nodes_pole_and_twig():
    G = make_pole_and_twig()
    node = G.vs.select(name=0)[0]
    assert len(sc.get_terminal_nodes(node, G)) == 2
    assert set(sc.get_terminal_nodes(node, G)) == set([1000,999])

def test_get_terminal_nodes_1_tree():
    G = make_depth_1_tree()
    node = G.vs.select(name=0)[0]
    assert len(sc.get_terminal_nodes(node, G)) == 1000

if __name__ == "__main__":
    test_get_node_breakdown_threshold()
    test_get_terminal_nodes()
