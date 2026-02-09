from _typeshed import Incomplete
from igraph._igraph import ADJ_DIRECTED as ADJ_DIRECTED, ADJ_LOWER as ADJ_LOWER, ADJ_MAX as ADJ_MAX, ADJ_MIN as ADJ_MIN, ADJ_PLUS as ADJ_PLUS, ADJ_UNDIRECTED as ADJ_UNDIRECTED, ADJ_UPPER as ADJ_UPPER, ALL as ALL, ARPACKOptions as ARPACKOptions, BFSIter as BFSIter, BLISS_F as BLISS_F, BLISS_FL as BLISS_FL, BLISS_FLM as BLISS_FLM, BLISS_FM as BLISS_FM, BLISS_FS as BLISS_FS, BLISS_FSM as BLISS_FSM, DFSIter as DFSIter, Edge as Edge, GET_ADJACENCY_BOTH as GET_ADJACENCY_BOTH, GET_ADJACENCY_LOWER as GET_ADJACENCY_LOWER, GET_ADJACENCY_UPPER as GET_ADJACENCY_UPPER, GraphBase as GraphBase, IN as IN, InternalError as InternalError, OUT as OUT, REWIRING_SIMPLE as REWIRING_SIMPLE, REWIRING_SIMPLE_LOOPS as REWIRING_SIMPLE_LOOPS, STAR_IN as STAR_IN, STAR_MUTUAL as STAR_MUTUAL, STAR_OUT as STAR_OUT, STAR_UNDIRECTED as STAR_UNDIRECTED, STRONG as STRONG, TRANSITIVITY_NAN as TRANSITIVITY_NAN, TRANSITIVITY_ZERO as TRANSITIVITY_ZERO, TREE_IN as TREE_IN, TREE_OUT as TREE_OUT, TREE_UNDIRECTED as TREE_UNDIRECTED, Vertex as Vertex, WEAK as WEAK, __igraph_version__ as __igraph_version__, arpack_options as default_arpack_options, community_to_membership as community_to_membership, convex_hull as convex_hull, is_bigraphical as is_bigraphical, is_degree_sequence as is_degree_sequence, is_graphical as is_graphical, is_graphical_degree_sequence as is_graphical_degree_sequence, set_progress_handler as set_progress_handler, set_random_number_generator as set_random_number_generator, set_status_handler as set_status_handler, umap_compute_weights as umap_compute_weights
from igraph.clustering import Clustering as Clustering, CohesiveBlocks as CohesiveBlocks, Cover as Cover, Dendrogram as Dendrogram, VertexClustering as VertexClustering, VertexCover as VertexCover, VertexDendrogram as VertexDendrogram, compare_communities as compare_communities, split_join_distance as split_join_distance
from igraph.configuration import Configuration as Configuration
from igraph.cut import Cut as Cut, Flow as Flow
from igraph.datatypes import DyadCensus as DyadCensus, Matrix as Matrix, TriadCensus as TriadCensus, UniqueIdGenerator as UniqueIdGenerator
from igraph.drawing import BoundingBox as BoundingBox, CairoGraphDrawer as CairoGraphDrawer, DefaultGraphDrawer as DefaultGraphDrawer, MatplotlibGraphDrawer as MatplotlibGraphDrawer, Plot as Plot, Point as Point, Rectangle as Rectangle, plot as plot
from igraph.drawing.colors import AdvancedGradientPalette as AdvancedGradientPalette, ClusterColoringPalette as ClusterColoringPalette, GradientPalette as GradientPalette, Palette as Palette, PrecalculatedPalette as PrecalculatedPalette, RainbowPalette as RainbowPalette, color_name_to_rgb as color_name_to_rgb, color_name_to_rgba as color_name_to_rgba, hsl_to_rgb as hsl_to_rgb, hsla_to_rgba as hsla_to_rgba, hsv_to_rgb as hsv_to_rgb, hsva_to_rgba as hsva_to_rgba, known_colors as known_colors, palettes as palettes, rgb_to_hsl as rgb_to_hsl, rgb_to_hsv as rgb_to_hsv, rgba_to_hsla as rgba_to_hsla, rgba_to_hsva as rgba_to_hsva
from igraph.drawing.utils import autocurve as autocurve
from igraph.layout import Layout as Layout
from igraph.matching import Matching as Matching
from igraph.operators import disjoint_union as disjoint_union, intersection as intersection, union as union
from igraph.seq import EdgeSeq as EdgeSeq, VertexSeq as VertexSeq
from igraph.statistics import FittedPowerLaw as FittedPowerLaw, Histogram as Histogram, RunningMean as RunningMean, mean as mean, median as median, percentile as percentile, power_law_fit as power_law_fit, quantile as quantile
from igraph.summary import GraphSummary as GraphSummary, summary as summary
from igraph.utils import rescale as rescale
from igraph.version import __version__ as __version__, __version_info__ as __version_info__

__all__ = ['config', 'AdvancedGradientPalette', 'BoundingBox', 'CairoGraphDrawer', 'ClusterColoringPalette', 'Clustering', 'CohesiveBlocks', 'Configuration', 'Cover', 'Cut', 'DefaultGraphDrawer', 'Dendrogram', 'DyadCensus', 'Edge', 'EdgeSeq', 'FittedPowerLaw', 'Flow', 'GradientPalette', 'Graph', 'GraphBase', 'GraphSummary', 'Histogram', 'InternalError', 'Layout', 'Matching', 'MatplotlibGraphDrawer', 'Matrix', 'Palette', 'Plot', 'Point', 'PrecalculatedPalette', 'RainbowPalette', 'Rectangle', 'RunningMean', 'TriadCensus', 'UniqueIdGenerator', 'Vertex', 'VertexClustering', 'VertexCover', 'VertexDendrogram', 'VertexSeq', 'autocurve', 'color_name_to_rgb', 'color_name_to_rgba', 'community_to_membership', 'compare_communities', 'convex_hull', 'default_arpack_options', 'disjoint_union', 'get_include', 'hsla_to_rgba', 'hsl_to_rgb', 'hsva_to_rgba', 'hsv_to_rgb', 'is_bigraphical', 'is_degree_sequence', 'is_graphical', 'is_graphical_degree_sequence', 'intersection', 'known_colors', 'load', 'mean', 'median', 'palettes', 'percentile', 'plot', 'power_law_fit', 'quantile', 'read', 'rescale', 'rgba_to_hsla', 'rgb_to_hsl', 'rgba_to_hsva', 'rgb_to_hsv', 'save', 'set_progress_handler', 'set_random_number_generator', 'set_status_handler', 'split_join_distance', 'summary', 'umap_compute_weights', 'union', 'write', '__igraph_version__', '__version__', '__version_info__', 'ADJ_DIRECTED', 'ADJ_LOWER', 'ADJ_MAX', 'ADJ_MIN', 'ADJ_PLUS', 'ADJ_UNDIRECTED', 'ADJ_UPPER', 'ALL', 'ARPACKOptions', 'BFSIter', 'BLISS_F', 'BLISS_FL', 'BLISS_FLM', 'BLISS_FM', 'BLISS_FS', 'BLISS_FSM', 'DFSIter', 'GET_ADJACENCY_BOTH', 'GET_ADJACENCY_LOWER', 'GET_ADJACENCY_UPPER', 'IN', 'OUT', 'REWIRING_SIMPLE', 'REWIRING_SIMPLE_LOOPS', 'STAR_IN', 'STAR_MUTUAL', 'STAR_OUT', 'STAR_UNDIRECTED', 'STRONG', 'TRANSITIVITY_NAN', 'TRANSITIVITY_ZERO', 'TREE_IN', 'TREE_OUT', 'TREE_UNDIRECTED', 'WEAK']

class Graph(GraphBase):
    omega: Incomplete
    alpha: Incomplete
    shell_index: Incomplete
    cut_vertices: Incomplete
    blocks: Incomplete
    evcent: Incomplete
    vertex_disjoint_paths: Incomplete
    edge_disjoint_paths: Incomplete
    cohesion: Incomplete
    adhesion: Incomplete
    shortest_paths: Incomplete
    shortest_paths_dijkstra = shortest_paths
    subgraph: Incomplete
    def __init__(self, *args, **kwds) -> None: ...
    from_networkx: Incomplete
    to_networkx: Incomplete
    from_graph_tool: Incomplete
    to_graph_tool: Incomplete
    Read_DIMACS: Incomplete
    write_dimacs: Incomplete
    Read_GraphMLz: Incomplete
    write_graphmlz: Incomplete
    Read_Pickle: Incomplete
    write_pickle: Incomplete
    Read_Picklez: Incomplete
    write_picklez: Incomplete
    Read_Adjacency: Incomplete
    write_adjacency: Incomplete
    write_svg: Incomplete
    Read: Incomplete
    Load = Read
    write: Incomplete
    save = write
    DictList: Incomplete
    to_dict_list: Incomplete
    TupleList: Incomplete
    to_tuple_list: Incomplete
    ListDict: Incomplete
    to_list_dict: Incomplete
    DictDict: Incomplete
    to_dict_dict: Incomplete
    Adjacency: Incomplete
    Weighted_Adjacency: Incomplete
    DataFrame: Incomplete
    get_vertex_dataframe: Incomplete
    get_edge_dataframe: Incomplete
    Bipartite: Incomplete
    Biadjacency: Incomplete
    Full_Bipartite: Incomplete
    Random_Bipartite: Incomplete
    GRG: Incomplete
    Formula: Incomplete
    def summary(self, verbosity: int = 0, width=None, *args, **kwds): ...
    def is_named(self): ...
    def is_weighted(self): ...
    @property
    def vs(self): ...
    @property
    def es(self): ...
    add_edge: Incomplete
    add_edges: Incomplete
    add_vertex: Incomplete
    add_vertices: Incomplete
    delete_edges: Incomplete
    clear: Incomplete
    as_directed: Incomplete
    as_undirected: Incomplete
    __iadd__: Incomplete
    __add__: Incomplete
    __and__: Incomplete
    __isub__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __or__: Incomplete
    disjoint_union: Incomplete
    union: Incomplete
    intersection: Incomplete
    get_adjacency: Incomplete
    get_adjacency_sparse: Incomplete
    get_adjlist: Incomplete
    get_biadjacency: Incomplete
    get_inclist: Incomplete
    indegree: Incomplete
    outdegree: Incomplete
    degree_distribution: Incomplete
    pagerank: Incomplete
    all_st_cuts: Incomplete
    all_st_mincuts: Incomplete
    gomory_hu_tree: Incomplete
    maxflow: Incomplete
    mincut: Incomplete
    st_mincut: Incomplete
    biconnected_components: Incomplete
    clusters: Incomplete
    cohesive_blocks: Incomplete
    connected_components: Incomplete
    components: Incomplete
    community_fastgreedy: Incomplete
    community_infomap: Incomplete
    community_leading_eigenvector: Incomplete
    community_label_propagation: Incomplete
    community_multilevel: Incomplete
    community_optimal_modularity: Incomplete
    community_edge_betweenness: Incomplete
    community_spinglass: Incomplete
    community_walktrap: Incomplete
    k_core: Incomplete
    community_leiden: Incomplete
    modularity: Incomplete
    layout: Incomplete
    layout_auto: Incomplete
    layout_sugiyama: Incomplete
    __plot__: Incomplete
    maximum_bipartite_matching: Incomplete
    bipartite_projection: Incomplete
    bipartite_projection_size: Incomplete
    count_automorphisms_vf2: Incomplete
    get_automorphisms_vf2: Incomplete
    def get_all_simple_paths(self, v, to=None, cutoff: int = -1, mode: str = 'out'): ...
    def path_length_hist(self, directed: bool = True): ...
    def dfs(self, vid, mode=...): ...
    def spanning_tree(self, weights=None, return_tree: bool = True): ...
    def dyad_census(self, *args, **kwds): ...
    def triad_census(self, *args, **kwds): ...
    def transitivity_avglocal_undirected(self, mode: str = 'nan', weights=None): ...
    def __bool__(self) -> bool: ...
    def __coerce__(self, other): ...
    def __reduce__(self): ...
    __iter__: Incomplete
    __hash__: Incomplete
    @classmethod
    def Incidence(cls, *args, **kwds): ...
    def are_connected(self, *args, **kwds): ...
    def get_incidence(self, *args, **kwds): ...

def get_include(): ...
def read(filename, *args, **kwds): ...
load = read

def write(graph, filename, *args, **kwds): ...
save = write
config: Configuration
