from typing import Optional

from graphical_models import PAG, DAG, arrow_head_types as Mark


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _endpoint_marks(edge_type: str) -> tuple[Optional[str], Optional[str]]:
    """
    Return PAG/DAG endpoint marks for a pair ordered as (i, j).
    """
    endpoint_map = {
        'directed_ij': ('tail', 'arrow'),
        'directed_ji': ('arrow', 'tail'),
        'undirected': ('tail', 'tail'),
        'bidirected': ('arrow', 'arrow'),
        'circle': ('circle', 'circle'),
        'partial_ij': ('circle', 'arrow'),
        'partial_ji': ('arrow', 'circle'),
        'partial_ij_tail': ('tail', 'circle'),
        'partial_ji_tail': ('circle', 'tail'),
        'none': (None, None),
    }
    return endpoint_map.get(edge_type, (None, None))


def _count_matching_endpoint_marks(reference_edge_type: str, predicted_edge_type: str) -> tuple[int, int, int]:
    """
    Count exact endpoint-mark matches for one unordered node pair.

    Returns:
        (correct_matches, reference_endpoint_total, predicted_endpoint_total)
    """
    reference_marks = _endpoint_marks(reference_edge_type)
    predicted_marks = _endpoint_marks(predicted_edge_type)
    correct_matches = sum(
        1
        for reference_mark, predicted_mark in zip(reference_marks, predicted_marks)
        if reference_mark is not None and predicted_mark == reference_mark
    )
    reference_total = sum(1 for mark in reference_marks if mark is not None)
    predicted_total = sum(1 for mark in predicted_marks if mark is not None)
    return correct_matches, reference_total, predicted_total

def dag_to_pag_comparison(dag: DAG, pag: PAG) -> dict:
    """
    Compare DAG and PAG by evaluating adjacencies and orientations.

    Args:
        dag: The ground truth DAG
        pag: The learned PAG graph

    Returns:
        Dictionary with detailed comparison metrics
    """
    nodes = pag.nodes_set.union(dag.nodes_set)

    metrics = {
        'missing_edges': 0,
        'extra_edges': 0,
        'correct_edges': 0,
        'orientation_errors': 0,
        'partial_orientations': 0,
        'correct_orientations': 0,
        'shd_strict': 0,
        'shd_partial': 0,
        'shd_edges_only': 0,
        'edge_precision': 0.0,
        'edge_recall': 0.0,
        'edge_f1': 0.0,
        'endpoint_precision': 0.0,
        'endpoint_recall': 0.0,
        'endpoint_f1': 0.0,
    }

    total_oriented_edges = 0
    correctly_oriented_edges = 0
    interventional_agreements = 0
    total_interventional_pairs = 0
    correct_endpoint_marks = 0
    total_reference_endpoint_marks = 0
    total_predicted_endpoint_marks = 0

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            pag_adjacent = pag.is_connected(i, j)
            dag_adjacent = dag.is_connected(i, j)
            pag_edge_type = _get_pag_edge_type(pag, i, j) if pag_adjacent else 'none'
            dag_edge_type = _get_dag_edge_type(dag, i, j) if dag_adjacent else 'none'
            endpoint_matches, reference_endpoint_total, predicted_endpoint_total = _count_matching_endpoint_marks(
                dag_edge_type,
                pag_edge_type,
            )
            correct_endpoint_marks += endpoint_matches
            total_reference_endpoint_marks += reference_endpoint_total
            total_predicted_endpoint_marks += predicted_endpoint_total

            if dag_adjacent and not pag_adjacent:
                metrics['missing_edges'] += 1
                metrics['shd_strict'] += 1
                metrics['shd_partial'] += 1
                metrics['shd_edges_only'] += 1
            elif pag_adjacent and not dag_adjacent:
                metrics['extra_edges'] += 1
                metrics['shd_strict'] += 1
                metrics['shd_partial'] += 1
                metrics['shd_edges_only'] += 1
            elif pag_adjacent and dag_adjacent:
                metrics['correct_edges'] += 1

                status = _orientations_compatible_dag(dag_edge_type, pag_edge_type)
                if status == 'compatible':
                    metrics['correct_orientations'] += 1
                elif status == 'partial':
                    metrics['partial_orientations'] += 1
                    metrics['shd_strict'] += 1
                else:
                    metrics['orientation_errors'] += 1
                    metrics['shd_strict'] += 1
                    metrics['shd_partial'] += 1

                # Orientation accuracy: matches on fully directed reference edges only.
                # Partial/ambiguous predictions are counted as incorrect for this metric.
                if dag_edge_type in ['directed_ij', 'directed_ji']:
                    total_oriented_edges += 1
                    if pag_edge_type == dag_edge_type:
                        correctly_oriented_edges += 1

            # Interventional Agreement: Check if causal effects match
            # i -> j in DAG means intervention on i affects j
            if dag_edge_type == 'directed_ij':
                total_interventional_pairs += 1
                # PAG should capture this as i -> j or i o-> j (partial)
                if pag_edge_type == 'directed_ij':
                    interventional_agreements += 1
                elif pag_edge_type in ['partial_ij', 'partial_ij_tail']:
                    # Partial agreement - edge exists but not fully oriented
                    interventional_agreements += 0.5
            elif dag_edge_type == 'directed_ji':
                total_interventional_pairs += 1
                if pag_edge_type == 'directed_ji':
                    interventional_agreements += 1
                elif pag_edge_type in ['partial_ji', 'partial_ji_tail']:
                    interventional_agreements += 0.5

    # Calculate percentages
    precision = _safe_divide(metrics['correct_edges'], metrics['correct_edges'] + metrics['extra_edges'])
    recall = _safe_divide(metrics['correct_edges'], metrics['correct_edges'] + metrics['missing_edges'])
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    metrics['edge_precision'] = precision
    metrics['edge_recall'] = recall
    metrics['edge_f1'] = f1

    endpoint_precision = _safe_divide(correct_endpoint_marks, total_predicted_endpoint_marks)
    endpoint_recall = _safe_divide(correct_endpoint_marks, total_reference_endpoint_marks)
    metrics['endpoint_precision'] = endpoint_precision
    metrics['endpoint_recall'] = endpoint_recall
    metrics['endpoint_f1'] = _safe_divide(2 * endpoint_precision * endpoint_recall, endpoint_precision + endpoint_recall)
    metrics['orientation_accuracy'] = _safe_divide(correctly_oriented_edges, total_oriented_edges)
    metrics['interventional_agreement'] = _safe_divide(interventional_agreements, total_interventional_pairs)

    return metrics


def _get_pag_edge_type(pag: PAG, i: int, j: int) -> str:
    """
    Determine PAG edge type between nodes i and j.
    Returns: 'directed_ij', 'directed_ji', 'undirected', 'bidirected', 'circle',
             'partial_ij', 'partial_ji', 'partial_ij_tail', 'partial_ji_tail', 'none'
    """
    # Directed edges
    if pag.is_parent(i, j):
        return 'directed_ij'
    if pag.is_parent(j, i):
        return 'directed_ji'
    # Bidirected edge
    if i in pag._graph[j][Mark.Directed] and j in pag._graph[i][Mark.Directed]:
        return 'bidirected' # should never happen
    # Undirected edge (tail-tail)
    if i in pag._graph[j][Mark.Tail] and j in pag._graph[i][Mark.Tail]:
        return 'undirected' # should never happen
    # Circle edge
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Circle]:
        return 'circle'
    # Partial edges
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Directed]:
        return 'partial_ji'
    if j in pag._graph[i][Mark.Circle] and i in pag._graph[j][Mark.Directed]:
        return 'partial_ij'
    if i in pag._graph[j][Mark.Circle] and j in pag._graph[i][Mark.Tail]:
        return 'partial_ij_tail'
    if j in pag._graph[i][Mark.Circle] and i in pag._graph[j][Mark.Tail]:
        return 'partial_ji_tail'
    return 'none'


def _get_dag_edge_type(dag: DAG, i: int, j: int) -> str:
    """
    Determine DAG edge type between nodes i and j.
    Returns: 'directed_ij', 'directed_ji', 'undirected', 'none'
    """
    if i in dag.parents(j):
        return 'directed_ij'
    if j in dag.parents(i):
        return 'directed_ji'
    if dag.is_connected(i, j) and dag.is_connected(j, i):
        return 'undirected' # should never happen
    return 'none'

def _orientations_compatible_dag(dag_type: str, pag_type: str) -> str:
    """
    Check if PAG and DAG edge orientations are compatible.
    Returns: 'compatible', 'partial', or 'incompatible'
    """
    if pag_type == dag_type:
        return 'compatible'
    if pag_type in ['partial_ij', 'partial_ji', 'partial_ij_tail', 'partial_ji_tail', 'circle', 'undirected']:
        return 'partial'
    if pag_type == 'bidirected':
        return 'incompatible'
    return 'incompatible'

def _orientations_compatible_pag(ref_type: str, pag_type: str) -> str:
    """
    Check if PAG and reference PAG edge orientations are compatible.
    Returns: 'compatible', 'partial', or 'incompatible'
    """
    if ref_type == pag_type:
        return 'compatible'

    ref_partial = ref_type in ['partial_ij', 'partial_ji', 'partial_ij_tail', 'partial_ji_tail', 'circle', 'undirected']
    pag_partial = pag_type in ['partial_ij', 'partial_ji', 'partial_ij_tail', 'partial_ji_tail', 'circle', 'undirected']

    if ref_partial or pag_partial:
        return 'partial'

    directed_pairs = {
        ('directed_ij', 'directed_ji'),
        ('directed_ji', 'directed_ij'),
    }
    if (ref_type, pag_type) in directed_pairs:
        return 'incompatible'

    if ref_type == 'bidirected' or pag_type == 'bidirected':
        return 'incompatible'

    return 'incompatible'

def pag_to_pag_comparison(reference_pag: PAG, pag: PAG) -> dict:
    """
    Compare PAG and reference PAG by evaluating adjacencies and orientations.

    Args:
        reference_pag: The reference PAG graph
        pag: The PAG graph to compare against the reference

    Returns:
        Dictionary with detailed comparison metrics
    """
    assert isinstance(reference_pag, PAG)
    assert isinstance(pag, PAG)

    nodes = pag.nodes_set.union(reference_pag.nodes_set)

    metrics = {
        'missing_edges': 0,
        'extra_edges': 0,
        'correct_edges': 0,
        'orientation_errors': 0,
        'partial_orientations': 0,
        'correct_orientations': 0,
        'shd_strict': 0,
        'shd_partial': 0,
        'shd_edges_only': 0,
        'edge_precision': 0.0,
        'edge_recall': 0.0,
        'edge_f1': 0.0,
        'endpoint_precision': 0.0,
        'endpoint_recall': 0.0,
        'endpoint_f1': 0.0,
    }

    total_oriented_edges = 0
    correctly_oriented_edges = 0
    interventional_agreements = 0
    total_interventional_pairs = 0
    correct_endpoint_marks = 0
    total_reference_endpoint_marks = 0
    total_predicted_endpoint_marks = 0

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            ref_adjacent = reference_pag.is_connected(i, j)
            pag_adjacent = pag.is_connected(i, j)
            ref_edge_type = _get_pag_edge_type(reference_pag, i, j) if ref_adjacent else 'none'
            pag_edge_type = _get_pag_edge_type(pag, i, j) if pag_adjacent else 'none'
            endpoint_matches, reference_endpoint_total, predicted_endpoint_total = _count_matching_endpoint_marks(
                ref_edge_type,
                pag_edge_type,
            )
            correct_endpoint_marks += endpoint_matches
            total_reference_endpoint_marks += reference_endpoint_total
            total_predicted_endpoint_marks += predicted_endpoint_total

            if ref_adjacent and not pag_adjacent:
                metrics['missing_edges'] += 1
                metrics['shd_strict'] += 1
                metrics['shd_partial'] += 1
                metrics['shd_edges_only'] += 1
            elif pag_adjacent and not ref_adjacent:
                metrics['extra_edges'] += 1
                metrics['shd_strict'] += 1
                metrics['shd_partial'] += 1
                metrics['shd_edges_only'] += 1
            elif pag_adjacent and ref_adjacent:
                metrics['correct_edges'] += 1

                status = _orientations_compatible_pag(ref_edge_type, pag_edge_type)
                if status == 'compatible':
                    metrics['correct_orientations'] += 1
                elif status == 'partial':
                    metrics['partial_orientations'] += 1
                    metrics['shd_strict'] += 1
                else:
                    metrics['orientation_errors'] += 1
                    metrics['shd_strict'] += 1
                    metrics['shd_partial'] += 1

                # Orientation accuracy: matches on fully directed reference edges only.
                # Partial/ambiguous predictions are counted as incorrect for this metric.
                if ref_edge_type in ['directed_ij', 'directed_ji']:
                    total_oriented_edges += 1
                    if pag_edge_type == ref_edge_type:
                        correctly_oriented_edges += 1

            if ref_edge_type == 'directed_ij':
                total_interventional_pairs += 1
                if pag_edge_type == 'directed_ij':
                    interventional_agreements += 1
                elif pag_edge_type in ['partial_ij', 'partial_ij_tail']:
                    interventional_agreements += 0.5
            elif ref_edge_type == 'directed_ji':
                total_interventional_pairs += 1
                if pag_edge_type == 'directed_ji':
                    interventional_agreements += 1
                elif pag_edge_type in ['partial_ji', 'partial_ji_tail']:
                    interventional_agreements += 0.5

    precision = _safe_divide(metrics['correct_edges'], metrics['correct_edges'] + metrics['extra_edges'])
    recall = _safe_divide(metrics['correct_edges'], metrics['correct_edges'] + metrics['missing_edges'])
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    metrics['edge_precision'] = precision
    metrics['edge_recall'] = recall
    metrics['edge_f1'] = f1

    endpoint_precision = _safe_divide(correct_endpoint_marks, total_predicted_endpoint_marks)
    endpoint_recall = _safe_divide(correct_endpoint_marks, total_reference_endpoint_marks)
    metrics['endpoint_precision'] = endpoint_precision
    metrics['endpoint_recall'] = endpoint_recall
    metrics['endpoint_f1'] = _safe_divide(2 * endpoint_precision * endpoint_recall, endpoint_precision + endpoint_recall)
    metrics['orientation_accuracy'] = _safe_divide(correctly_oriented_edges, total_oriented_edges)
    metrics['interventional_agreement'] = _safe_divide(interventional_agreements, total_interventional_pairs)

    return metrics


def compare_edge_details(dag: DAG, pag: PAG, max_display: int = 10) -> tuple[list[str], list[str]]:
    """
    Print detailed comparison of edges between PAG and DAG, with partial orientations in a separate section.
    """
    assert isinstance(dag, DAG)
    assert isinstance(pag, PAG)

    nodes = sorted(pag.nodes_set.union(dag.nodes_set))
    differences = []
    partials = []

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            pag_connected = pag.is_connected(i, j)
            dag_connected = dag.is_connected(i, j) or dag.is_connected(j, i)

            if not pag_connected and not dag_connected:
                continue

            pag_type = _get_pag_edge_type(pag, i, j) if pag_connected else 'none'
            dag_type = _get_dag_edge_type(dag, i, j) if dag_connected else 'none'

            if not pag_connected:
                differences.append(f"Missing: {i} --- {j} (DAG: {dag_type})")
            elif not dag_connected:
                differences.append(f"Extra: {i} --- {j} (PAG: {pag_type})")
            else:
                status = _orientations_compatible_dag(dag_type, pag_type)
                if status == 'incompatible':
                    differences.append(f"Wrong orientation: {i} --- {j} (DAG: {dag_type}, PAG: {pag_type})")
                elif status == 'partial':
                    partials.append(f"Partial orientation: {i} --- {j} (DAG: {dag_type}, PAG: {pag_type})")

    if differences:
        print(f"Found {len(differences)} differences:")
        for idx, diff in enumerate(differences[:max_display]):
            print(f"  {idx + 1}. {diff}")
        if len(differences) > max_display:
            print(f"  ... and {len(differences) - max_display} more")
    else:
        print("No differences found - graphs match perfectly!")

    if partials:
        print(f"Found {len(partials)} partial orientation differences:")
        for idx, diff in enumerate(partials[:max_display]):
            print(f"  {idx + 1}. {diff}")
        if len(partials) > max_display:
            print(f"  ... and {len(partials) - max_display} more")

    return differences, partials

def compare_pag_edge_details(reference_pag: PAG, pag: PAG, max_display: int = 10) -> tuple[list[str], list[str]]:
    """
    Print detailed comparison of edges between two PAGs, with partial orientations in a separate section.
    """
    assert isinstance(reference_pag, PAG)
    assert isinstance(pag, PAG)

    nodes = sorted(pag.nodes_set.union(reference_pag.nodes_set))
    differences = []
    partials = []

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            ref_connected = reference_pag.is_connected(i, j)
            pag_connected = pag.is_connected(i, j)

            if not ref_connected and not pag_connected:
                continue

            ref_type = _get_pag_edge_type(reference_pag, i, j) if ref_connected else 'none'
            pag_type = _get_pag_edge_type(pag, i, j) if pag_connected else 'none'

            if not pag_connected:
                differences.append(f"Missing: {i} --- {j} (Reference PAG: {ref_type})")
            elif not ref_connected:
                differences.append(f"Extra: {i} --- {j} (PAG: {pag_type})")
            else:
                status = _orientations_compatible_pag(ref_type, pag_type)
                if status == 'incompatible':
                    differences.append(f"Wrong orientation: {i} --- {j} (Reference PAG: {ref_type}, PAG: {pag_type})")
                elif status == 'partial':
                    partials.append(f"Partial orientation: {i} --- {j} (Reference PAG: {ref_type}, PAG: {pag_type})")

    if differences:
        print(f"Found {len(differences)} differences:")
        for idx, diff in enumerate(differences[:max_display]):
            print(f"  {idx + 1}. {diff}")
        if len(differences) > max_display:
            print(f"  ... and {len(differences) - max_display} more")
    else:
        print("No differences found - graphs match perfectly!")

    if partials:
        print(f"Found {len(partials)} partial orientation differences:")
        for idx, diff in enumerate(partials[:max_display]):
            print(f"  {idx + 1}. {diff}")
        if len(partials) > max_display:
            print(f"  ... and {len(partials) - max_display} more")

    return differences, partials
