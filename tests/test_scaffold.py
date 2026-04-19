"""Milestone 0: verify the package installs and is importable."""


def test_package_importable():
    import valuefunctionethics

    assert valuefunctionethics.__version__ == "0.1.0"


def test_submodules_importable():
    from valuefunctionethics import envs, rewards, agents, utils

    assert envs is not None
    assert rewards is not None
    assert agents is not None
    assert utils is not None
