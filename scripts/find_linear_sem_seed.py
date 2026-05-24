#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from data.linear_sem_benchmark import find_first_valid_linear_sem_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the first Linear SEM seed whose random connected DAG satisfies the benchmark validity rule."
    )
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, required=True)
    parser.add_argument("--expected-neighborhood-size", type=int, required=True)
    parser.add_argument("--num-dags-timeout", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format for the accepted seed payload.",
    )
    return parser


def _render_text(payload: dict) -> str:
    benchmark = payload["benchmark"]
    lines = [
        f"accepted_seed: {payload['seed']}",
        f"edge_list: {payload['edge_list']}",
    ]
    for output in benchmark["outputs"]:
        lines.append(
            "output_node={output_node} ancestor_inputs={ancestor_inputs} non_ancestor_inputs={non_ancestor_inputs}".format(
                output_node=output["output_node"],
                ancestor_inputs=output["ancestor_inputs"],
                non_ancestor_inputs=output["non_ancestor_inputs"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    result = find_first_valid_linear_sem_seed(
        num_nodes=args.num_nodes,
        input_size=args.input_size,
        output_size=args.output_size,
        expected_neighborhood_size=args.expected_neighborhood_size,
        num_dags_timeout=args.num_dags_timeout,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    if result is None:
        message = {
            "error": "No valid Linear SEM seed found in the requested range.",
            "num_nodes": args.num_nodes,
            "input_size": args.input_size,
            "output_size": args.output_size,
            "expected_neighborhood_size": args.expected_neighborhood_size,
            "num_dags_timeout": args.num_dags_timeout,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
        }
        if args.format == "json":
            print(json.dumps(message, indent=2, sort_keys=True))
        else:
            print(message["error"], file=sys.stderr)
        return 1

    payload = result.to_dict()
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
