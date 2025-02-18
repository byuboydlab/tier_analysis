import tier_analysis as ta
import igraph as ig

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


if __name__ == "__main__":
    test_get_reachable_nodes()
