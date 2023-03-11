def k_core(G, k=1, fname="k_core.gml"):
    # next step -- k-core
    c = np.array(G.coreness())
    core_ind = list(np.where(c > k)[0])
    G = G.induced_subgraph(core_ind)
    save(G, fname)
    return G, core_ind
