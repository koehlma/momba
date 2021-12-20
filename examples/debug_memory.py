import pathlib

from momba.debug import memory

memory.dump_statistics(pathlib.Path("memory-statistics.txt"))
