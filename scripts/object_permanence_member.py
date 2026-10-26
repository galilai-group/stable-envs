"""Member identity and paths only; no source execution or rendering algorithms.

``member`` is a zero-based index into task.json's ``member_scripts``, not its
global ``members`` numbers. ``nominal`` means that selected script's defaults;
it does not mean member zero. Diversity profiles require a separate audit.
Candidate exporters and independent oracles may share this naming convention,
but must keep their preparation recipes and geometry extraction independent.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Member:
    task: str
    member: int = 0
    profile: str = "nominal"

    def __post_init__(self):
        if not isinstance(self.task, str) or re.fullmatch(r"G(?:0[1-9]|[1-9][0-9]{1,2})", self.task) is None:
            raise ValueError("task must use the source spelling, e.g. G02 or G103")
        if type(self.member) is not int or self.member < 0:
            raise ValueError("member must be a nonnegative integer script index")
        if self.profile != "nominal":
            raise ValueError("only the audited nominal profile is supported; diversity needs an explicit audit")

    @property
    def leaf(self):
        return "nominal" if self.member == 0 else f"member{self.member}"

    @property
    def clip(self):
        return f"{self.task.lower()}-{self.leaf}"

    def as_dict(self):
        return {"task": self.task, "member": self.member, "profile": self.profile}

    def check_source(self, source):
        """Accept historical implicit member-zero metadata without rewriting it."""
        if (
            source["task"] != self.task
            or type(source.get("member", 0)) is not int
            or source.get("member", 0) != self.member
            or source.get("entry_profile", "nominal") != self.profile
        ):
            raise ValueError("source task/member/profile identity differs")
