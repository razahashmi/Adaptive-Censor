"""Bounded, concurrent single-flight cache for decoded evidence and embeddings."""
from collections import OrderedDict
from concurrent.futures import Future
import threading


def size_of(value):
    if hasattr(value, 'nbytes'):
        return value.nbytes
    if isinstance(value, dict):
        return sum(size_of(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(size_of(v) for v in value)
    return 64


class BoundedCache:
    def __init__(self, maximum_bytes=128 * 1024**2):
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.values, self.pending = OrderedDict(), {}
        self.lock = threading.RLock()
        self.hits = self.misses = 0

    def get(self, key, loader):
        with self.lock:
            if key in self.values:
                value, size = self.values.pop(key)
                self.values[key] = (value, size)
                self.hits += 1
                return value
            owner = key not in self.pending
            if owner:
                self.pending[key] = Future()
                self.misses += 1
            else:
                self.hits += 1
            future = self.pending[key]
        if not owner:
            return future.result()
        try:
            value = loader()
            size = size_of(value)
            with self.lock:
                if size <= self.maximum_bytes:
                    while self.values and self.bytes + size > self.maximum_bytes:
                        _, (_, old_size) = self.values.popitem(last=False)
                        self.bytes -= old_size
                    self.values[key] = (value, size)
                    self.bytes += size
                future.set_result(value)
            return value
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def get_many(self, keys, loader):
        """Batch only missing values; coalesce overlaps with other callers."""
        futures, owned = [], []
        with self.lock:
            for i, key in enumerate(keys):
                if key in self.values:
                    value,size = self.values.pop(key)
                    self.values[key] = (value,size)
                    future=Future(); future.set_result(value)
                    self.hits += 1
                elif key in self.pending:
                    future=self.pending[key]
                    self.hits += 1
                else:
                    future=Future(); self.pending[key]=future
                    owned.append((i,key,future)); self.misses += 1
                futures.append(future)
        if owned:
            try:
                values=loader([i for i,_,_ in owned])
                if len(values)!=len(owned):
                    raise ValueError('Batch loader returned the wrong number of values')
                with self.lock:
                    for (_,key,future),value in zip(owned,values):
                        size=size_of(value)
                        if size<=self.maximum_bytes:
                            while self.values and self.bytes+size>self.maximum_bytes:
                                _,(_,old_size)=self.values.popitem(last=False)
                                self.bytes -= old_size
                            self.values[key]=(value,size); self.bytes += size
                        future.set_result(value)
            except BaseException as error:
                for _,_,future in owned:
                    if not future.done(): future.set_exception(error)
                raise
            finally:
                with self.lock:
                    for _,key,_ in owned: self.pending.pop(key,None)
        return [future.result() for future in futures]


def source_key(path):
    from pathlib import Path
    path = Path(path)
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns
