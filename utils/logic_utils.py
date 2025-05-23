from typing import List

class Ops(object):
    AND: str = "AND"
    OR: str = "OR"
    NOT: str = "NOT"

def process_atoms(
    atoms: List[str],
    ops: List[str]
    ):
    """
    Categorize and group atoms by the type of set operations attached to it. 
    """
    assert len(atoms) == len(ops) + 1, "The number of operations should be 1 less than the number of atoms"

    # Default first operation
    # The first operation is OR only if all other operations are OR
    ops_set = set(ops)
    if len(ops_set) > 0 and Ops.OR in ops_set:
        ops = [Ops.OR] + ops  # Default first operation to AND
    else:
        ops = [Ops.AND] + ops

    and_atoms = set()
    or_atoms = set()
    not_atoms = set()

    for atom, op in zip(atoms, ops):
        if op == Ops.AND:
            and_atoms.add(atom)
        elif op == Ops.OR:
            or_atoms.add(atom)
        elif op == Ops.NOT:
            not_atoms.add(atom)
    return and_atoms, or_atoms, not_atoms

def is_subset(
    superset_atoms: List[str],
    superset_ops: List[str],
    subset_atoms: List[str],
    subset_ops: List[str]
    ) -> bool:
    """
    Check if two queries are subset of each other.

    Idea: Check that all of the following is true.

    - The super sets' Intersect/AND terms should be a subset of the subset's UNION/AND terms
    - The super sets' Union/OR terms should be a superset of the subset's Union/OR terms
    - The super sets' NOT terms should be a subset of the subset's NOT terms

    Note that this function assumes that the query makes sense by itself.
    e.g. There are no invalid cases like "A and B not B". 
    """

    superset_and_atoms, superset_or_atoms, superset_not_atoms = process_atoms(superset_atoms, superset_ops)
    subset_and_atoms, subset_or_atoms, subset_not_atoms = process_atoms(subset_atoms, subset_ops)

    and_constraint = superset_and_atoms.issubset(subset_and_atoms)
    or_constraint = superset_or_atoms.issuperset(subset_or_atoms)
    not_constraint = superset_not_atoms.issubset(subset_not_atoms)
    
    return and_constraint and or_constraint and not_constraint

def is_strictly_exclusive(
    left_atoms: List[str], 
    left_ops: List[str],
    right_atoms: List[str],
    right_ops: List[str]) -> bool:
    """
    Check if two queries are strictly mutually exclusive or not, 
    i.e. There is no way that the results from two queries overlap.

    - Any of the AND terms of the right set are found in the NOT terms of the left set.
    - All of the OR terms of the right set are found in the NOT terms of the left set.
    """
    
    left_and_atoms, left_or_atoms, left_not_atoms = process_atoms(left_atoms, left_ops)
    right_and_atoms, right_or_atoms, right_not_atoms = process_atoms(right_atoms, right_ops)

    left_exclusive_right_and = not right_and_atoms.isdisjoint(left_not_atoms)
    left_exclusive_right_or = right_or_atoms.issubset(left_not_atoms)

    right_exclusive_left_and = not left_and_atoms.isdisjoint(right_not_atoms)
    right_exclusive_left_or = left_or_atoms.issubset(right_not_atoms)

    return (left_exclusive_right_and and left_exclusive_right_or) or (right_exclusive_left_and and right_exclusive_left_or)

def is_loosely_exclusive(
    left_atoms: List[str], 
    left_ops: List[str],
    right_atoms: List[str],
    right_ops: List[str]) -> bool:
    """
    Check if two queries are mutually exclusive, assuming any two atomic queries are disjoint.

    Under this assumption, we only need to check if the `OR` `AND` atoms between the two queries are disjoint.

    - If the `OR` `AND` atoms of the two queries have any intersection, the queries are not exclusive.
    - If the `OR` `AND` atoms are disjoint, the queries are exclusive.
    """

    left_and_atoms, left_or_atoms, left_not_atoms = process_atoms(left_atoms, left_ops)
    right_and_atoms, right_or_atoms, right_not_atoms = process_atoms(right_atoms, right_ops)

    if not left_and_atoms.isdisjoint(right_not_atoms):
        left_and_atoms = set()
    
    if not right_and_atoms.isdisjoint(left_not_atoms):
        right_and_atoms = set()

    left_inclusive_atoms = (left_and_atoms | left_or_atoms) - right_not_atoms 
    right_inclusive_atoms = (right_and_atoms | right_or_atoms) - left_not_atoms

    return left_inclusive_atoms.isdisjoint(right_inclusive_atoms)