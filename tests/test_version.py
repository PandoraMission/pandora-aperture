# First-party/Local
from pandoraaperture import __version__


def test_example():
    # This is an example test, you should write your own.
    # Note that above you have imported from your package *as it is installed*
    # Do not import functions relative to the test directory.

    assert __version__ == "0.1.0"
