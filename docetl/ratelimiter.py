import math
from inspect import isawaitable
from typing import Any

import pyrate_limiter


class BucketCollection(pyrate_limiter.BucketFactory):
    def __init__(self, **buckets: dict[str, pyrate_limiter.InMemoryBucket]) -> None:
        self.clock = pyrate_limiter.TimeClock()
        self.buckets = buckets

    def wrap_item(self, name: str, weight: int = 1) -> pyrate_limiter.RateItem:
        now = self.clock.now()

        async def wrap_async():
            return pyrate_limiter.RateItem(name, await now, weight=weight)

        def wrap_sync():
            return pyrate_limiter.RateItem(name, now, weight=weight)

        return wrap_async() if isawaitable(now) else wrap_sync()

    def get(self, item: pyrate_limiter.RateItem) -> pyrate_limiter.AbstractBucket:
        if item.name not in self.buckets:
            return self.buckets["unknown"]
        return self.buckets[item.name]


def create_bucket_factory(rate_limits: dict[str, Any]) -> BucketCollection:
    """
    Create a BucketCollection from rate limits configuration.

    Args:
        rate_limits: Dictionary containing rate limit configuration

    Returns:
        BucketCollection configured with the specified rate limits
    """
    buckets = {
        param: pyrate_limiter.InMemoryBucket(
            [
                pyrate_limiter.Rate(
                    param_limit["count"],
                    param_limit["per"]
                    * getattr(
                        pyrate_limiter.Duration,
                        param_limit.get("unit", "SECOND").upper(),
                    ),
                )
                for param_limit in param_limits
            ]
        )
        for param, param_limits in rate_limits.items()
    }

    # Add default bucket for unknown parameters
    buckets["unknown"] = pyrate_limiter.InMemoryBucket(
        [pyrate_limiter.Rate(math.inf, 1)]
    )

    return BucketCollection(**buckets)
