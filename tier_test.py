import tier_analysis as ta
import igraph as ig
import copy


# Utilties

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


# Tests

def test_get_reachable_nodes():
    depth_1_tree = make_depth_1_tree()
    for v in depth_1_tree.vs:
        if v["Tier"] == 0:
            assert len(ta.get_reachable_nodes(v, depth_1_tree)) == len(depth_1_tree.vs)
        elif v["Tier"] == 1:
            assert len(ta.get_reachable_nodes(v, depth_1_tree)) == 1
        else:
            raise Exception("Something is wrong with the tree!")
        
    depth_2_tree = make_depth_2_tree()
    for v in depth_2_tree.vs:
        if v["Tier"] == 0:
            assert len(ta.get_reachable_nodes(v, depth_2_tree)) == len(depth_2_tree.vs)
        elif v["Tier"] == 1:
            assert len(ta.get_reachable_nodes(v, depth_2_tree)) == 11
        elif v["Tier"] == 2:
            assert len(ta.get_reachable_nodes(v, depth_2_tree)) == 1
        else:
            raise Exception("Something is wrong with the tree!")
        
    pole_and_twig = make_pole_and_twig()
    for v in pole_and_twig.vs:
        if v.index == 0:
            assert len(ta.get_reachable_nodes(v, pole_and_twig)) == len(pole_and_twig.vs)
        elif v.index == 1000:
            assert len(ta.get_reachable_nodes(v, pole_and_twig)) == 1
        else:
            assert len(ta.get_reachable_nodes(v, pole_and_twig)) == len(pole_and_twig.vs) - v.index - 1
        

def test_get_terminal_nodes():
    depth_1_tree = make_depth_1_tree()
    for v in depth_1_tree.vs:
        if v["Tier"] == 0:
            assert len(ta.get_terminal_nodes(v, depth_1_tree)) == 1000
        elif v["Tier"] == 1:
            assert len(ta.get_terminal_nodes(v, depth_1_tree)) == 1
        else:
            raise Exception("Something is wrong with the tree!")
        
    depth_2_tree = make_depth_2_tree()
    for v in depth_2_tree.vs:
        if v["Tier"] == 0:
            assert len(ta.get_terminal_nodes(v, depth_2_tree)) == 10000
        elif v["Tier"] == 1:
            assert len(ta.get_terminal_nodes(v, depth_2_tree)) == 10
        elif v["Tier"] == 2:
            assert len(ta.get_terminal_nodes(v, depth_2_tree)) == 1
        else:
            raise Exception("Something is wrong with the tree!")
        
    pole_and_twig = make_pole_and_twig()
    for v in pole_and_twig.vs:
        if v.index == 0:
            assert len(ta.get_terminal_nodes(v, pole_and_twig)) == 2
        else:
            assert len(ta.get_terminal_nodes(v, pole_and_twig)) == 1


def test_get_upstream():
    depth_1_tree = make_depth_1_tree()
    assert len(ta.get_upstream(depth_1_tree.vs[0], depth_1_tree, depth_1_tree)) == 1001

    depth_1_tree_thin = copy.deepcopy(depth_1_tree)

    for i in range(1,501):
        depth_1_tree_thin.delete_vertices(depth_1_tree_thin.vs.find(name=i).index)
        assert len(ta.get_upstream(depth_1_tree.vs[0], depth_1_tree, depth_1_tree_thin)) == 1001 - i

    depth_1_tree_thin.delete_vertices(0)
    assert len(ta.get_upstream(depth_1_tree.vs[0], depth_1_tree, depth_1_tree_thin)) == 0

    depth_2_tree = make_depth_2_tree()
    assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree)) == 11001
    assert len(ta.get_upstream(depth_2_tree.vs[1000], depth_2_tree, depth_2_tree)) == 11

    depth_2_tree_thin = copy.deepcopy(depth_2_tree)
    depth_2_tree_thin.delete_vertices(depth_2_tree_thin.vs.find(name=11000).index)
    assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree_thin)) == 11000
    assert len(ta.get_upstream(depth_2_tree.vs[1000], depth_2_tree, depth_2_tree_thin)) == 10

    for i in range(10999, 10990):
        print(i)
        depth_2_tree_thin.delete_vertices(depth_2_tree_thin.vs.find(name=i).index)
        assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree_thin)) == 11000 - (i - 10990)
        assert len(ta.get_upstream(depth_2_tree.vs[1000], depth_2_tree, depth_2_tree_thin)) == 10 - (i - 10990)

    depth_2_tree_thin.delete_vertices(1000)
    assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree_thin)) == 10990
    assert len(ta.get_upstream(depth_2_tree.vs[1000], depth_2_tree, depth_2_tree_thin)) == 0

    for i in range(900, 1000):
        depth_2_tree_thin.delete_vertices(depth_2_tree_thin.vs.find(name=i).index)
        assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree_thin)) == 10990 - ((i - 899) * 11)
        assert len(ta.get_upstream(depth_2_tree.vs[i], depth_2_tree, depth_2_tree_thin)) == 0

    depth_2_tree_thin.delete_vertices(0)
    assert len(ta.get_upstream(depth_2_tree.vs[0], depth_2_tree, depth_2_tree_thin)) == 0

    pole_and_twig = make_pole_and_twig()
    assert len(ta.get_upstream(pole_and_twig.vs[0], pole_and_twig, pole_and_twig)) == 1001

    pole_and_twig_thin = copy.deepcopy(pole_and_twig)

    pole_and_twig_thin.delete_vertices(1)
    assert len(ta.get_upstream(pole_and_twig.vs[0], pole_and_twig, pole_and_twig_thin)) == 2

    pole_and_twig_thin.delete_vertices(0)
    assert len(ta.get_upstream(pole_and_twig.vs[0], pole_and_twig, pole_and_twig_thin)) == 0


def test_some_terminal_suppliers_reachable():
    pass


def test_percent_terminal_suppliers_reachable():
    pass


if __name__ == "__main__":
    test_get_reachable_nodes()
    test_get_terminal_nodes()
    test_get_upstream()
