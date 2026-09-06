#!/usr/bin/env python3
"""Compare precision between two benchmark outputs using cosine similarity."""
import sys
import json
import math
import re


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def mean_absolute_error(a, b):
    """Compute mean absolute error between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return float('inf')
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def parse_output_vectors(filepath):
    """Parse output vectors from a benchmark log file.

    Supports both JSON_RESULT lines with an 'output' field
    and raw space/comma-separated numbers in [ ... ] brackets.
    """
    with open(filepath) as f:
        content = f.read()

    # Try JSON_RESULT output vectors first
    vectors = []
    for line in content.split('\n'):
        if line.startswith('JSON_RESULT: '):
            try:
                data = json.loads(line[13:])
                output = data.get('output') or data.get('outputs')
                if output is not None:
                    vectors.append(output)
            except json.JSONDecodeError:
                pass

    if vectors:
        return vectors

    # Fallback: parse [number number ...] arrays
    array_matches = re.findall(r'\[\s*([\d.\-eE+\s,]+)\]', content)
    for match in array_matches:
        try:
            vec = [float(x) for x in re.split(r'[\s,]+', match.strip()) if x]
            if len(vec) > 10:  # heuristic: real output vectors are large
                vectors.append(vec)
        except ValueError:
            pass

    return vectors


def compare_logs(truth_path, candidate_path):
    """Compare output precision between a truth (reference) log and a candidate log."""
    truth_vectors = parse_output_vectors(truth_path)
    candidate_vectors = parse_output_vectors(candidate_path)

    if not truth_vectors:
        print(f"Error: no output vectors found in truth file: {truth_path}", file=sys.stderr)
        return False
    if not candidate_vectors:
        print(f"Error: no output vectors found in candidate file: {candidate_path}", file=sys.stderr)
        return False

    print("=" * 60)
    print("Precision Comparison Report")
    print("=" * 60)
    print(f"Reference : {truth_path}")
    print(f"Candidate: {candidate_path}")
    print()

    vector_pairs = min(len(truth_vectors), len(candidate_vectors))
    print(f"Comparing {vector_pairs} output vector(s):")
    print()

    all_cosine = []
    all_mae = []

    for i in range(vector_pairs):
        tv = truth_vectors[i]
        cv = candidate_vectors[i]

        cs = cosine_similarity(tv, cv)
        mae = mean_absolute_error(tv, cv)

        all_cosine.append(cs)
        all_mae.append(mae)

        status = "PASS" if cs >= 0.99 else "WARN" if cs >= 0.95 else "FAIL"
        print(f"  Vector[{i}]:")
        print(f"    Cosine Similarity : {cs:.8f}  [{status}]")
        print(f"    Mean Abs Error    : {mae:.8f}")
        print(f"    Reference dims    : {len(tv)}")
        print(f"    Candidate dims    : {len(cv)}")
        print()

    avg_cosine = sum(all_cosine) / len(all_cosine)
    avg_mae = sum(all_mae) / len(all_mae)
    min_cosine = min(all_cosine)
    max_mae = max(all_mae)

    print("-" * 60)
    print(f"Summary:")
    print(f"  Avg Cosine Similarity : {avg_cosine:.8f}")
    print(f"  Min Cosine Similarity : {min_cosine:.8f}")
    print(f"  Avg Mean Abs Error    : {avg_mae:.8f}")
    print(f"  Max Mean Abs Error    : {max_mae:.8f}")
    print()

    if min_cosine >= 0.99:
        print("Result: PASS (cosine similarity >= 0.99 for all vectors)")
        return True
    elif min_cosine >= 0.95:
        print("Result: WARN (cosine similarity >= 0.95 but < 0.99)")
        return True
    else:
        print("Result: FAIL (cosine similarity < 0.95)")
        return True


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_precision.py <reference_log> <candidate_log>")
        print()
        print("Compares output precision between two benchmark log files.")
        print("Reference log should be from ONNX Runtime (ground truth).")
        print("Candidate log is from the backend under test.")
        sys.exit(1)

    truth_path = sys.argv[1]
    candidate_path = sys.argv[2]

    if not os.path.exists(truth_path):
        print(f"Error: file not found: {truth_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(candidate_path):
        print(f"Error: file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    compare_logs(truth_path, candidate_path)


if __name__ == '__main__':
    import os
    main()
