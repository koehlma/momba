import typing as t

import asyncio

from momba import gym


def oracle(state: t.Sequence[float], available: t.Sequence[bool]) -> int:
    print("State:", state)
    print("Available:", available)
    return 0


async def main():
    server = gym.checker.OracleServer(oracle)
    await server.start()
    print("Port:", server.port)
    await asyncio.sleep(30 * 60)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
