from secure_aggregation.node.dropout import DropoutManager, DropoutStage


def test_plan_is_deterministic_across_instances() -> None:
    participants = [f"node_{i}" for i in range(6)]
    manager_a = DropoutManager(participants, per_round=3, seed=99)
    manager_b = DropoutManager(participants, per_round=3, seed=99)

    plan_a = manager_a.plan_for_round(2)
    plan_b = manager_b.plan_for_round(2)

    assert plan_a.stage_map == plan_b.stage_map
    # Spot check that plan is stable for a given node
    node_id = participants[0]
    assert manager_a.stage_for(node_id, 2) == manager_b.stage_for(node_id, 2)


def test_plan_never_selects_more_than_requested() -> None:
    participants = ["u1", "u2"]
    manager = DropoutManager(participants, per_round=5, seed=7)

    plan = manager.plan_for_round(0)
    assert len(plan.stage_map) == len(participants)
    for stage in plan.stage_map.values():
        assert stage in (DropoutStage.BEFORE_ROUND0, DropoutStage.BEFORE_MASKED_INPUT)
