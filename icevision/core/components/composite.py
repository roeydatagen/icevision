__all__ = ["Component", "Composite"]

from icevision.imports import *


class Component:
    order = 0.5

    def __init__(self, composite):
        self.composite = composite


class Composite:
    base_components = set()

    def __init__(self, components):
        components = set(components).union(self.base_components)
        components = set(component(composite=self) for component in components)
        self._sort_components(components)

    def _sort_components(self, components):
        self.components = sorted(components, key=lambda o: o.order)
        self.components_cls = [comp.__class__ for comp in self.components]

    def __getattr__(self, name):
        # avoid recursion https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        if name == "components":
            raise AttributeError(name)
        # delegates attributes to components
        for component in self.components:
            try:
                return getattr(component, name)
            except AttributeError:
                pass
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def reduce_on_components(
        self, fn_name: str, reduction: Optional[str] = None, **fn_kwargs
    ) -> Any:
        results = []
        for component in self.components:
            results.append(getattr(component, fn_name)(**fn_kwargs))

        if reduction is not None and len(results) > 0:
            out = results.pop(0)
            for r in results:
                getattr(out, reduction)(r)
        else:
            out = results

        return out

    def get_component_by_type(self, component_type) -> Union[Component, None]:
        for component in self.components:
            if isinstance(component, component_type):
                return component

    def add_component(self, component):
        components = set(self.components)
        components.add(component(composite=self))
        self._sort_components(components)
