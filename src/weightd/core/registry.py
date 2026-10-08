from weightd.core.bundle import WeightBundle
from weightd.errors import BundleExistsError, BundleNotFoundError


class BundleRegistry:
    """
    A registry for managing WeightBundle instances.
    """

    def __init__(self):
        self._bundles: dict[str, WeightBundle] = {}

    def register(self, bundle: WeightBundle) -> None:
        if bundle.name in self._bundles:
            raise BundleExistsError(f"Bundle already exists: {bundle.name}")
        self._bundles[bundle.name] = bundle

    def get(self, name: str) -> WeightBundle:
        if name not in self._bundles:
            raise BundleNotFoundError(f"Bundle not found: {name}")
        return self._bundles[name]

    # def unregister(self, name: str) -> None:
    #     if name not in self._bundles:
    #         raise KeyError(f"No bundle registered with name '{name}'.")
    #     del self._bundles[name]

    def list(self):
        # equivalent to list(self._bundles.keys())
        return list(self._bundles)
