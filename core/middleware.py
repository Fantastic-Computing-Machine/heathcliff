# ABOUTME: Coordinator execution limits.


class DelegationBudget:
    """Enforce coordinator budget limits on delegation depth and task count.

    This is not a LangChain middleware subclass — it is used directly by the
    coordinator graph to check limits before dispatching subtasks.
    """

    def __init__(
        self,
        max_tasks_per_request: int = 10,
        per_task_timeout_ms: int = 60000,
        max_total_runtime_ms: int = 300000,
    ) -> None:
        self.max_tasks_per_request = max_tasks_per_request
        self.per_task_timeout_ms = per_task_timeout_ms
        self.max_total_runtime_ms = max_total_runtime_ms

    def check_task_count(self, count: int) -> None:
        """Raise if task count exceeds budget."""
        if count > self.max_tasks_per_request:
            raise ValueError(
                f"Task count {count} exceeds max_tasks_per_request "
                f"({self.max_tasks_per_request})"
            )


def create_execution_budget() -> DelegationBudget:
    """Create coordinator execution limits from Config defaults."""
    from config import Config

    return DelegationBudget(
        max_tasks_per_request=Config.MAX_TASKS_PER_REQUEST,
        per_task_timeout_ms=Config.PER_TASK_TIMEOUT_MS,
        max_total_runtime_ms=Config.MAX_TOTAL_RUNTIME_MS,
    )
