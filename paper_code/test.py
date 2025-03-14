import sc
import pharma_analysis as pa
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt


def make_depth_1_tree():
    G = ig.Graph(directed=True)
    G.add_vertex()
    vcount = 1000
    G.add_vertices(vcount)
    G.add_edges([(i, 0) for i in range(1, vcount + 1)])
    G.vs['name'] = list(range(G.vcount()))  # helps when passing to subgraphs
    G.vs['Tier'] = 1
    G.vs[0]['Tier'] = 0

    return G


def make_depth_2_tree():
    G = make_depth_1_tree()
    degree = 10

    node_counter = G.vcount()
    for node in G.vs.select(Tier=1):
        G.add_vertices(
            degree,
            attributes={
                'name': list(
                    range(
                        node_counter,
                        node_counter +
                        degree))})
        G.add_edges([(i, node.index)
                    for i in range(node_counter, node_counter + degree)])
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
    G.add_edges([(i + 1, i) for i in range(999)])
    G.vs['name'] = list(range(G.vcount()))  # helps when passing to subgraphs
    # assign each edge to a tier based on distance from the root
    G.es['Tier'] = [i for i in range(999)]
    # add an additional node to give the leaf indegree 2
    G.add_vertex()
    G.add_edge(1000, 0)
    G.vs[1000]['name'] = 1000
    return G


def test_get_node_breakdown_threshold_pole_and_twig():
    G = make_pole_and_twig()
    threshold = pa.get_node_breakdown_threshold(
        G.vs.select(
            name=0)[0],
        G,
        breakdown_threshold=.95,
        thinning_ratio=1 /
        1000)
    assert (threshold == 1)


def test_get_node_breakdown_threshold_tree_1():
    # make a graph that has one node and 1000 neighbors
    G = make_depth_1_tree()
    node = G.vs.select(name=0)[0]

    # make sure the breakdown threshold is within 1 of 50 for a .95 threshold
    # need standard deviation of binomial distribution
    thinning_ratio = 5 / 1000
    sd = np.sqrt(G.vcount() * thinning_ratio * (1 - thinning_ratio))
    assert abs(
        pa.get_node_breakdown_threshold(
            node,
            G,
            breakdown_threshold=.95,
            thinning_ratio=thinning_ratio) - 50) <= 2 * sd


def test_get_node_breakdown_threshold_tree_2():
    # make a graph that has one node and 1000 neighbors
    G = make_depth_2_tree()
    node = G.vs.select(name=0)[0]

    # make sure the breakdown threshold is within 1 of 50 for a .95 threshold
    # need standard deviation of binomial distribution
    thinning_ratio = 5 / 1000
    sd = np.sqrt(G.vcount() * thinning_ratio * (1 - thinning_ratio))
    # solve for the number of nodes that need to be deleted
    # assert abs(get_node_breakdown_threshold(node, G, breakdown_threshold=.95, thinning_ratio = thinning_ratio) - 50) <= 2*sd


def test_get_terminal_nodes():
    test_get_terminal_nodes_1_tree()
    test_get_terminal_nodes_pole_and_twig()


def test_get_terminal_nodes_pole_and_twig():
    G = make_pole_and_twig()
    node = G.vs.select(name=0)[0]
    assert len(sc.get_terminal_nodes(node, G)) == 2
    assert set(sc.get_terminal_nodes(node, G)) == set([1000, 999])


def test_get_terminal_nodes_1_tree():
    G = make_depth_1_tree()
    node = G.vs.select(name=0)[0]
    assert len(sc.get_terminal_nodes(node, G)) == 1000

def tree_graph_test():
    tree_depth = 5
    tree_count = 20
    repeats = 12
    degree = 4
    tol = .05

    N = int((degree**(tree_depth + 1) - 1) / (tree_depth - 1))

    Gs = []
    for i in range(tree_count):
        G = ig.Graph.Tree(N, degree, "in")
        for scale in ['country', 'industry', 'country-industry']:
            G.vs[scale] = ['foo'] * N
        G.vs['name'] = list(range(i * N, (i + 1) * N))
        Gs.append(G)
    G = ig.operators.union(Gs)

    med_suppliers = [i * N for i in range(tree_count)]

    res = sc.failure_reachability(G, med_suppliers=med_suppliers, repeats=repeats)

    gb = res.groupby('Percent firms remaining')
    observed = gb.mean()['Avg. percent end suppliers reachable']
    rho = gb.first().index
    expected = rho**(tree_depth + 1)
    plt.plot(rho, expected)

    abs_err = np.max(np.abs(observed - expected))
    if abs_err < tol:
        print("passed with error " + str(abs_err))
    else:
        print("failed with error " + str(abs_err))

    return res

if __name__ == "__main__":
    test_get_node_breakdown_threshold()
    test_get_terminal_nodes()
