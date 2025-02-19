from inspect import isawaitable

import pyrate_limiter


class BucketCollection(pyrate_limiter.BucketFactory):
    def __init__(self, **buckets):
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
