import stim

from stimsymb.multi_qubit import split_mpp_targets_into_products


def render_target(target: stim.GateTarget) -> str:
    if target.is_combiner:
        return "*"

    prefix = "X" if target.is_x_target else "Y" if target.is_y_target else "Z"
    inverted = "!" if target.is_inverted_result_target else ""
    qubit = target.qubit_value
    assert qubit is not None
    return f"{inverted}{prefix}{qubit}"


def describe_target(target: stim.GateTarget) -> dict[str, object]:
    return {
        "value": target.value,
        "qubit_value": target.qubit_value,
        "is_x_target": target.is_x_target,
        "is_y_target": target.is_y_target,
        "is_z_target": target.is_z_target,
        "is_combiner": target.is_combiner,
        "is_inverted_result_target": target.is_inverted_result_target,
    }


def main() -> None:
    circuit = stim.Circuit("MPP X0*Y1*Z2 X3*X4 !Z5*X6")
    instruction = list(circuit)[0]
    targets = instruction.targets_copy()
    products = split_mpp_targets_into_products(targets)

    print("Instruction:")
    print(f"  {instruction}")
    print("\nPretty view:")
    print("  flat:   " + " ".join(render_target(target) for target in targets))
    print(
        "  split:  "
        + " | ".join(" ".join(render_target(target) for target in product) for product in products)
    )

    print("\nFlat Stim targets:")
    for index, target in enumerate(targets):
        print(f"  {index:>2}: {render_target(target):<4} {describe_target(target)}")

    print("\nSplit MPP products:")
    for index, product in enumerate(products):
        print(f"  product {index}: {' '.join(render_target(target) for target in product)}")
        for target in product:
            print(f"    - {describe_target(target)}")


if __name__ == "__main__":
    main()
