import pkgutil
import importlib

packages = [
    "amt",
    "selfplay_chess",
    "ui",
    "examples.math_bin_hv.hypervector",
]


def _import_everything(pkg: str) -> None:
    if pkg.startswith("examples."):
        import os

        rel = pkg.split(".")[1:]
        path = os.path.join("examples", *rel) + ".py"
        spec = importlib.util.spec_from_file_location(pkg, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return

    module = importlib.import_module(pkg)
    mod_path = getattr(module, "__path__", None)
    if mod_path:
        for _, name, _ in pkgutil.walk_packages(mod_path, prefix=pkg + "."):
            importlib.import_module(name)


def test_import_all() -> None:
    for pkg in packages:
        _import_everything(pkg)
