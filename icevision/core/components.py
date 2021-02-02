__all__ = [
    "registry",
    "imageid",
    "classmap",
    "size",
    "filepath",
    "bbox",
    "label",
    "mask",
    "area",
    "iscrowd",
    "keypoint",
]

from icevision.imports import *


class _ComponentRegistry:
    def __init__(self):
        self.components = {}
        self.component2name = {}

    def new_component_registry(self, name):
        if name in self.components:
            raise ValueError("{name} is already registered")

        self.components[name] = []
        return partial(self.register_component, name=name)

    def register_component(self, component, name):
        self.components[name].append(component)
        self.component2name[component] = name
        return component

    def get_components_groups(self, components):
        names = []
        for component in components:
            try:
                names.append(self.component2name[component])
            except KeyError:
                pass
        return names

    def match_components(self, base_cls, components):
        """Matches components with the same type but for base_cls."""
        names = self.get_components_groups(components)
        return [
            comp
            for name in names
            for comp in self.components[name]
            if issubclass(comp, base_cls)
        ]


registry = _ComponentRegistry()

imageid = registry.new_component_registry("imageid")
classmap = registry.new_component_registry("classmap")
size = registry.new_component_registry("size")
filepath = registry.new_component_registry("filepath")
bbox = registry.new_component_registry("bbox")
label = registry.new_component_registry("label")
mask = registry.new_component_registry("mask")
area = registry.new_component_registry("area")
iscrowd = registry.new_component_registry("iscrowd")
keypoint = registry.new_component_registry("keypoint")
