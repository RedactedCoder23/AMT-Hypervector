import pkgutil
import importlib

packages = ["amt", "selfplay_chess", "ui"]


def test_import_all():
    for pkg in packages:
        module = importlib.import_module(pkg)
        for loader, name, is_pkg in pkgutil.walk_packages(
            module.__path__, prefix=pkg + "."
        ):
            importlib.import_module(name)
