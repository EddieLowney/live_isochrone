"""A tiny name-to-class registry.

Every pluggable role (grid, provider, sampler, reconstructor, store, ledger)
gets its own Registry. Implementations register themselves with a decorator and
the config file names them as strings, so adding a new sampling algorithm means
writing one class and changing one line of YAML.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, role: str) -> None:
        self.role = role
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def wrap(cls: type[T]) -> type[T]:
            key = name.lower()
            if key in self._entries:
                raise ValueError(f"{self.role} '{name}' is already registered")
            self._entries[key] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
            return cls

        return wrap

    def get(self, name: str) -> type[T]:
        key = name.lower()
        if key not in self._entries:
            known = ", ".join(sorted(self._entries)) or "none"
            raise KeyError(f"unknown {self.role} '{name}'. Registered: {known}")
        return self._entries[key]

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate by name, passing only the kwargs the class accepts. This
        lets one config block carry options for several alternative
        implementations without each one choking on the others' keys."""
        cls = self.get(name)
        import inspect

        if cls.__init__ is object.__init__:
            # No constructor of its own; object.__init__ claims to take
            # **kwargs but rejects them at runtime.
            return cls()  # type: ignore[call-arg]
        sig = inspect.signature(cls.__init__)
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            return cls(**kwargs)  # type: ignore[call-arg]
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return cls(**allowed)  # type: ignore[call-arg]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries


GRIDS: Registry = Registry("grid")
PROVIDERS: Registry = Registry("provider")
SAMPLERS: Registry = Registry("sampler")
RECONSTRUCTORS: Registry = Registry("reconstructor")
STORES: Registry = Registry("store")
LEDGERS: Registry = Registry("ledger")
PACERS: Registry = Registry("pacer")
