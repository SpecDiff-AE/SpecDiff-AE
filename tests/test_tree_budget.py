from diffspec.core.tree_budget import HistoryTreeBudgetController


def test_history_budget_stays_fixed_during_warmup():
    controller = HistoryTreeBudgetController(
        base_nodes=48,
        max_depth=8,
        window_size=10,
        max_nodes=80,
        round_to=4,
    )

    budgets = [controller.observe(3) for _ in range(10)]

    assert budgets == [48] * 10


def test_history_budget_grows_after_recent_window():
    controller = HistoryTreeBudgetController(
        base_nodes=48,
        max_depth=8,
        window_size=10,
        max_nodes=80,
        round_to=4,
    )

    for _ in range(10):
        controller.observe(3)
    next_budget = controller.observe(3)

    assert next_budget > 48
    assert next_budget <= 80
    assert next_budget % 4 == 0


def test_history_budget_is_monotone_after_warmup():
    controller = HistoryTreeBudgetController(
        base_nodes=48,
        max_depth=8,
        window_size=3,
        max_nodes=80,
        round_to=4,
    )

    budgets = [controller.observe(accept) for accept in [6, 5, 4, 7, 8, 3, 2, 5]]

    assert budgets[:3] == [48, 48, 48]
    assert budgets[3:] == sorted(budgets[3:])


def test_history_budget_clamps_to_configured_limit():
    controller = HistoryTreeBudgetController(
        base_nodes=48,
        max_depth=8,
        window_size=2,
        warmup_steps=0,
        max_nodes=56,
        round_to=4,
    )

    budgets = [controller.observe(1) for _ in range(8)]

    assert max(budgets) == 56
    assert all(48 <= budget <= 56 for budget in budgets)


def test_history_budget_statistics_are_serializable():
    controller = HistoryTreeBudgetController(base_nodes=32, max_depth=6, window_size=3, max_nodes=48)
    for accept in [2, 3, 4, 2]:
        controller.observe(accept)

    stats = controller.get_statistics()

    assert stats["policy"] == "late_ramp_history"
    assert stats["base_nodes"] == 32
    assert stats["max_nodes"] == 48
    assert stats["history"][-1]["accept_length"] == 2


def test_history_budget_from_env_uses_history_by_default(monkeypatch):
    monkeypatch.delenv("DIFFSPEC_TREE_BUDGET_POLICY", raising=False)

    controller = HistoryTreeBudgetController.from_env(base_nodes=48, max_depth=8)

    assert controller is not None
    assert controller.window_size == 10


def test_history_budget_from_env_overrides(monkeypatch):
    monkeypatch.setenv("DIFFSPEC_TREE_BUDGET_POLICY", "history")
    monkeypatch.setenv("DIFFSPEC_TREE_BUDGET_WINDOW", "5")
    monkeypatch.setenv("DIFFSPEC_TREE_BUDGET_MAX", "72")

    controller = HistoryTreeBudgetController.from_env(base_nodes=48, max_depth=8)

    assert controller is not None
    assert controller.window_size == 5
    assert controller.max_nodes == 72


def test_history_budget_from_env_fixed_opt_out(monkeypatch):
    monkeypatch.setenv("DIFFSPEC_TREE_BUDGET_POLICY", "fixed")

    assert HistoryTreeBudgetController.from_env(base_nodes=48, max_depth=8) is None
