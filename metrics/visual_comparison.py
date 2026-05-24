from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from graphical_models import PAG, DAG, arrow_head_types as Mark
from plot_utils.draw_graph import draw_edge, draw_node
from plot_utils.graph_layout import ForceDirectedLayout, CircleLayout

GRAPH_TITLE_FONTSIZE = 14

def _calc_layout(dag, all_nodes, node_radius, layout_type):
    factor = 1000
    default_layout = ForceDirectedLayout(dag, (-factor, factor), (-factor, factor), num_iterations=100)
    layout = default_layout if layout_type is None or layout_type == 'force' else CircleLayout(dag, (-factor, factor), (-factor, factor))
    nodes_pos = layout.calc_layout()
    for node in dag.nodes_set:
        nodes_pos[node] = nodes_pos[node] / factor
        nodes_pos[node] = nodes_pos[node] * (1 - 4 * node_radius)
        nodes_pos[node] = (nodes_pos[node] + 1) / 2
    for node in all_nodes:
        if node not in nodes_pos:
            nodes_pos[node] = np.random.rand(2) * 0.8 + 0.1
    return nodes_pos

def _draw_nodes(ax, all_nodes, nodes_pos, node_radius, node_labels, dag_nodes, latent_nodes, selection_nodes):
    for node in all_nodes:
        fill_color = '#ffcccc' if node not in dag_nodes else 'white'
        line_color = 'red' if node not in dag_nodes else 'black'
        contour = 'rectangle' if node in latent_nodes or node in selection_nodes else 'circle'
        draw_node(ax, nodes_pos[node], node_radius=node_radius,
                  node_name=node_labels[node], contour=contour,
                  line_color=line_color, fill_color=fill_color, text_color='black')

def _create_legend(fig):
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color='green', lw=2.5, label='Correct edge and direction (i→j)'),
        Line2D([0], [0], color='black', lw=2.5, label='Correct edge, incorrect or unresolved direction'),
        Line2D([0], [0], color='red', lw=2.5, label='False edge'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=3,
               fontsize=15,
               frameon=False,
               bbox_to_anchor=(0.5, 0.02))

def _create_pag_vs_pag_legend(fig):
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='green',  label='Edge matches reference PAG'),
        mpatches.Patch(color='black',  label='Edge exists but orientation partially known'),
        mpatches.Patch(color='orange', label='Edge exists but reversed direction'),
        mpatches.Patch(color='red',    label='Extra edge or incompatible'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=4,
               fontsize=10,
               frameon=False,
               bbox_to_anchor=(0.5, 0.02))

def _pag_directed_endmarks(pag: PAG, u, v):
    mu = pag.get_edge_mark(node_parent=v, node_child=u)
    mv = pag.get_edge_mark(node_parent=u, node_child=v)
    if mu == Mark.Tail and mv == Mark.Directed:
        return (u, v)
    if mu == Mark.Directed and mv == Mark.Tail:
        return (v, u)
    return None

def _edge_color_for_pag_pair(pag: PAG, u, v, dag):
    if not pag.is_connected(u, v):
        return None
    if u not in dag.nodes_set or v not in dag.nodes_set:
        return 'red'
    if not dag.is_connected(u, v) and not dag.is_connected(v, u):
        return 'red'

    # Treat all non-matching orientations as correct adjacency but incorrect direction.
    mu = pag.get_edge_mark(node_parent=v, node_child=u)
    mv = pag.get_edge_mark(node_parent=u, node_child=v)
    if mu == Mark.Directed and mv == Mark.Directed:
        return 'black'

    dir_pair = _pag_directed_endmarks(pag, u, v)
    if dir_pair is None:
        # Undirected or circle-ended edges
        return 'black'
    s, t = dir_pair
    if s in dag.parents(t):
        return 'green'
    elif t in dag.parents(s):
        return 'black'
    else:
        return 'red'

def _get_pag_edge_type(pag: PAG, i: int, j: int) -> str:
    if pag.is_parent(i, j):
        return 'directed_ij'
    if pag.is_parent(j, i):
        return 'directed_ji'
    if i in pag._graph[j][Mark.Directed] and j in pag._graph[i][Mark.Directed]:
        return 'bidirected'
    if i in pag._graph[j][Mark.Tail] and j in pag._graph[i][Mark.Tail]:
        return 'undirected'
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Circle]:
        return 'circle'
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Directed]:
        return 'partial_ji'
    if j in pag._graph[i][Mark.Circle] and i in pag._graph[j][Mark.Directed]:
        return 'partial_ij'
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Tail]:
        return 'partial_ij_tail'
    if j in pag._graph[i][Mark.Circle] and i in pag._graph[j][Mark.Tail]:
        return 'partial_ji_tail'
    return 'none'

def _edge_color_for_pag_vs_pag(pag: PAG, u, v, reference_pag: PAG):
    if not pag.is_connected(u, v):
        return None
    if not reference_pag.is_connected(u, v):
        return 'red'

    pag_type = _get_pag_edge_type(pag, u, v)
    ref_type = _get_pag_edge_type(reference_pag, u, v)

    if pag_type == ref_type:
        return 'green'

    partial_types = {
        'partial_ij', 'partial_ji', 'partial_ij_tail', 'partial_ji_tail', 'circle', 'undirected'
    }
    if pag_type in partial_types or ref_type in partial_types:
        return 'black'

    if (pag_type, ref_type) in {('directed_ij', 'directed_ji'), ('directed_ji', 'directed_ij')}:
        return 'orange'

    return 'red'

def compare_pags_against_dag(
        dag: DAG,
        pag_left: PAG,
        pag_right: PAG,
        *,
        node_labels: Optional[dict] = None,
        latent_nodes: Optional[set] = None,
        selection_nodes: Optional[set] = None,
        layout_type: Optional[str] = None,
        node_size_factor: float = 1.0,
        titles: tuple[str, str] = ("PAG 1 vs DAG", "PAG 2 vs DAG"),
        dag_title: str = "DAG",
        show: bool = True
    ):
    """
    Draw the DAG and two PAGs side-by-side (DAG first), with PAG edges colored by agreement with the DAG.
    """
    assert isinstance(dag, DAG)
    assert isinstance(pag_left, PAG)
    assert isinstance(pag_right, PAG)

    node_labels = node_labels or {}
    latent_nodes = latent_nodes or set()
    selection_nodes = selection_nodes or set()

    all_nodes = set(dag.nodes_set) | set(pag_left.nodes_set) | set(pag_right.nodes_set)
    for n in all_nodes:
        node_labels.setdefault(n, n)

    node_radius = 0.04 * node_size_factor
    nodes_pos = _calc_layout(dag, all_nodes, node_radius, layout_type)

    def _draw_dag_on_axes(ax):
        ax.set_axis_off()
        _draw_nodes(ax, dag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for child in dag.nodes_set:
            for parent in dag.parents(child):
                draw_edge(ax, nodes_pos[parent], nodes_pos[child], node_radius, line_color='black')
        ax.set_title(dag_title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    def _draw_pag_on_axes(ax, pag: PAG, title: str):
        ax.set_axis_off()
        _draw_nodes(ax, pag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for u, v in combinations(pag.nodes_set, 2):
            if not pag.is_connected(u, v):
                continue
            color = _edge_color_for_pag_pair(pag, u, v, dag)
            if color is None:
                continue
            mark_u = pag.get_edge_mark(node_parent=v, node_child=u)
            mark_v = pag.get_edge_mark(node_parent=u, node_child=v)
            draw_edge(ax, nodes_pos[u], nodes_pos[v], node_radius,
                      mark_u, mark_v, line_color=color, fill_color='white')
        ax.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    fig = plt.figure(figsize=(18, 6))
    ax_dag   = fig.add_axes((0.025, 0.08, 0.29, 0.84), frameon=False, aspect=1.)
    ax_left  = fig.add_axes((0.355, 0.08, 0.29, 0.84), frameon=False, aspect=1.)
    ax_right = fig.add_axes((0.685, 0.08, 0.29, 0.84), frameon=False, aspect=1.)

    _draw_dag_on_axes(ax_dag)
    _draw_pag_on_axes(ax_left,  pag_left,  titles[0])
    _draw_pag_on_axes(ax_right, pag_right, titles[1])

    _create_legend(fig)

    if show:
        plt.show()
        return None

    return fig

def compare_pags_against_pag(
        pag_left: PAG,
        pag_right: PAG,
        *,
        layout_graph: Optional[object] = None,
        node_labels: Optional[dict] = None,
        latent_nodes: Optional[set] = None,
        selection_nodes: Optional[set] = None,
        layout_type: Optional[str] = None,
        node_size_factor: float = 1.0,
        titles: tuple[str, str] = ("PAG 1 vs PAG 2", "PAG 2 vs PAG 1"),
        show: bool = True
    ):
    """
    Draw two PAGs side-by-side, with edges colored by agreement with the other PAG.
    """
    assert isinstance(pag_left, PAG)
    assert isinstance(pag_right, PAG)

    node_labels = node_labels or {}
    latent_nodes = latent_nodes or set()
    selection_nodes = selection_nodes or set()

    base_graph = layout_graph or pag_left
    all_nodes = set(pag_left.nodes_set) | set(pag_right.nodes_set)
    for n in all_nodes:
        node_labels.setdefault(n, n)

    node_radius = 0.04 * node_size_factor
    nodes_pos = _calc_layout(base_graph, all_nodes, node_radius, layout_type)

    def _draw_pag_on_axes(ax, pag: PAG, reference: PAG, title: str):
        ax.set_axis_off()
        _draw_nodes(ax, pag.nodes_set, nodes_pos, node_radius, node_labels, base_graph.nodes_set, latent_nodes, selection_nodes)
        for u, v in combinations(pag.nodes_set, 2):
            if not pag.is_connected(u, v):
                continue
            color = _edge_color_for_pag_vs_pag(pag, u, v, reference)
            if color is None:
                continue
            mark_u = pag.get_edge_mark(node_parent=v, node_child=u)
            mark_v = pag.get_edge_mark(node_parent=u, node_child=v)
            draw_edge(ax, nodes_pos[u], nodes_pos[v], node_radius,
                      mark_u, mark_v, line_color=color, fill_color='white')
        ax.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    fig = plt.figure(figsize=(12, 6))
    ax_left  = fig.add_axes((0.05, 0.08, 0.42, 0.84), frameon=False, aspect=1.)
    ax_right = fig.add_axes((0.53, 0.08, 0.42, 0.84), frameon=False, aspect=1.)

    _draw_pag_on_axes(ax_left,  pag_left,  pag_right, titles[0])
    _draw_pag_on_axes(ax_right, pag_right, pag_left,  titles[1])

    _create_pag_vs_pag_legend(fig)

    if show:
        plt.show()
        return None

    return fig

def compare_5pags_against_dag(
        dag: DAG,
        pag1: PAG,
        pag2: PAG,
        pag3: PAG,
        pag4: PAG,
        pag5: PAG,
        *,
        node_labels: Optional[dict] = None,
        latent_nodes: Optional[set] = None,
        selection_nodes: Optional[set] = None,
        layout_type: Optional[str] = None,
        node_size_factor: float = 1.0,
        titles: tuple[str, str, str, str, str] = ("PAG Fold 1", "PAG Fold 2", "PAG Fold 3", "PAG Fold 4", "PAG Fold 5"),
        dag_title: str = "DAG",
        show: bool = True
    ):
    """
    Draw the DAG and five PAGs in a 2-row layout:
    - First row: DAG, PAG1, PAG2
    - Second row: PAG3, PAG4, PAG5
    PAG edges are colored by agreement with the DAG.
    """
    assert isinstance(dag, DAG)
    assert isinstance(pag1, PAG)
    assert isinstance(pag2, PAG)
    assert isinstance(pag3, PAG)
    assert isinstance(pag4, PAG)
    assert isinstance(pag5, PAG)

    node_labels = node_labels or {}
    latent_nodes = latent_nodes or set()
    selection_nodes = selection_nodes or set()

    all_nodes = set(dag.nodes_set) | set(pag1.nodes_set) | set(pag2.nodes_set) | set(pag3.nodes_set) | set(pag4.nodes_set) | set(pag5.nodes_set)
    for n in all_nodes:
        node_labels.setdefault(n, n)

    node_radius = 0.04 * node_size_factor
    nodes_pos = _calc_layout(dag, all_nodes, node_radius, layout_type)

    def _draw_dag_on_axes(ax):
        ax.set_axis_off()
        _draw_nodes(ax, dag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for child in dag.nodes_set:
            for parent in dag.parents(child):
                draw_edge(ax, nodes_pos[parent], nodes_pos[child], node_radius, line_color='black')
        ax.set_title(dag_title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    def _draw_pag_on_axes(ax, pag: PAG, title: str):
        ax.set_axis_off()
        _draw_nodes(ax, pag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for u, v in combinations(pag.nodes_set, 2):
            if not pag.is_connected(u, v):
                continue
            color = _edge_color_for_pag_pair(pag, u, v, dag)
            if color is None:
                continue
            mark_u = pag.get_edge_mark(node_parent=v, node_child=u)
            mark_v = pag.get_edge_mark(node_parent=u, node_child=v)
            draw_edge(ax, nodes_pos[u], nodes_pos[v], node_radius,
                      mark_u, mark_v, line_color=color, fill_color='white')
        ax.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    fig = plt.figure(figsize=(13.5, 9.0))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.985,
        top=0.93,
        bottom=0.11,
        wspace=0.06,
        hspace=0.12,
    )

    ax_dag = fig.add_subplot(grid[0, 0], frameon=False, aspect=1.0)
    ax_pag1 = fig.add_subplot(grid[0, 1], frameon=False, aspect=1.0)
    ax_pag2 = fig.add_subplot(grid[0, 2], frameon=False, aspect=1.0)
    ax_pag3 = fig.add_subplot(grid[1, 0], frameon=False, aspect=1.0)
    ax_pag4 = fig.add_subplot(grid[1, 1], frameon=False, aspect=1.0)
    ax_pag5 = fig.add_subplot(grid[1, 2], frameon=False, aspect=1.0)

    _draw_dag_on_axes(ax_dag)
    _draw_pag_on_axes(ax_pag1, pag1, titles[0])
    _draw_pag_on_axes(ax_pag2, pag2, titles[1])
    _draw_pag_on_axes(ax_pag3, pag3, titles[2])
    _draw_pag_on_axes(ax_pag4, pag4, titles[3])
    _draw_pag_on_axes(ax_pag5, pag5, titles[4])

    _create_legend(fig)

    if show:
        plt.show()
        return None

    return fig

def compare_pag_against_dag(
        dag: DAG,
        pag: PAG,
        *,
        node_labels: Optional[dict] = None,
        latent_nodes: Optional[set] = None,
        selection_nodes: Optional[set] = None,
        layout_type: Optional[str] = None,
        node_size_factor: float = 1.0,
        titles: tuple[str, str] = ("PAG", "DAG"),
        show: bool = True
    ):
    """
    Draw the DAG and one PAG side-by-side (DAG first), with PAG edges colored by agreement with the DAG.
    """
    assert isinstance(dag, DAG)
    assert isinstance(pag, PAG)

    node_labels = node_labels or {}
    latent_nodes = latent_nodes or set()
    selection_nodes = selection_nodes or set()

    all_nodes = set(dag.nodes_set) | set(pag.nodes_set)
    for n in all_nodes:
        node_labels.setdefault(n, n)

    node_radius = 0.04 * node_size_factor
    nodes_pos = _calc_layout(dag, all_nodes, node_radius, layout_type)

    def _draw_dag_on_axes(ax):
        ax.set_axis_off()
        _draw_nodes(ax, dag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for child in dag.nodes_set:
            for parent in dag.parents(child):
                draw_edge(ax, nodes_pos[parent], nodes_pos[child], node_radius, line_color='black')
        ax.set_title(titles[1], fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    def _draw_pag_on_axes(ax, pag: PAG, title: str):
        ax.set_axis_off()
        _draw_nodes(ax, pag.nodes_set, nodes_pos, node_radius, node_labels, dag.nodes_set, latent_nodes, selection_nodes)
        for u, v in combinations(pag.nodes_set, 2):
            if not pag.is_connected(u, v):
                continue
            color = _edge_color_for_pag_pair(pag, u, v, dag)
            if color is None:
                continue
            mark_u = pag.get_edge_mark(node_parent=v, node_child=u)
            mark_v = pag.get_edge_mark(node_parent=u, node_child=v)
            draw_edge(ax, nodes_pos[u], nodes_pos[v], node_radius,
                      mark_u, mark_v, line_color=color, fill_color='white')
        ax.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE, fontweight='bold')

    fig = plt.figure(figsize=(12, 6))
    ax_dag   = fig.add_axes((0.05, 0.08, 0.42, 0.84), frameon=False, aspect=1.)
    ax_pag   = fig.add_axes((0.53, 0.08, 0.42, 0.84), frameon=False, aspect=1.)

    _draw_dag_on_axes(ax_dag)
    _draw_pag_on_axes(ax_pag, pag, titles[0])

    _create_legend(fig)

    if show:
        plt.show()
        return None

    return fig
