from collections import *        # Useful for deque, Counter, defaultdict, namedtuple, etc.
from itertools import *          # Provides tools for combinations, permutations, product, etc.
from functools import *          # Includes tools like lru_cache for memoization, reduce, etc.
from heapq import *              # Provides heap operations like heappush, heappop, useful for priority queues
from bisect import *             # For efficient binary search and maintaining sorted lists
from math import *               # Includes functions like gcd, sqrt, factorial, isqrt, comb, etc.
from operator import *           # Includes functions like itemgetter, attrgetter, add, mul for functional programming
from array import *              # For efficient storage and manipulation of numeric arrays
from typing import *             # Provides typing hints (List, Tuple, Dict, etc.) to improve readability and error-checking
from decimal import *            # High-precision arithmetic operations, useful for certain precision tasks
from queue import *              # Includes Queue, LifoQueue, PriorityQueue useful in BFS, DFS, and other algorithms
import sys
from typing import Generic, Iterable, Iterator, List, Tuple, TypeVar, Optional
T = TypeVar('T')
class SortedList(Generic[T]):
    BUCKET_RATIO = 16
    SPLIT_RATIO = 24
    def __init__(self, a: Iterable[T] = []) -> None:
        a = list(a)
        n = self.size = len(a)
        if any(a[i] > a[i + 1] for i in range(n - 1)):
            a.sort()
        num_bucket = int(ceil(sqrt(n / self.BUCKET_RATIO)))
        self.a = [a[n * i // num_bucket : n * (i + 1) // num_bucket] for i in range(num_bucket)]
    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j
    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    def __eq__(self, other) -> bool:
        return list(self) == list(other)
    def __len__(self) -> int:
        return self.size
    def __repr__(self) -> str:
        return "SortedMultiset" + str(self.a)
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"
    def _position(self, x: T) -> Tuple[List[T], int, int]:
        for i, a in enumerate(self.a):
            if x <= a[-1]: break
        return (a, i, bisect_left(a, x))
    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a, _, i = self._position(x)
        return i != len(a) and a[i] == x
    def count(self, x: T) -> int:
        return self.index_right(x) - self.index(x)
    def insert(self, x: T) -> None:
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a, b, i = self._position(x)
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.SPLIT_RATIO:
            mid = len(a) >> 1
            self.a[b:b+1] = [a[:mid], a[mid:]]
    def _pop(self, a: List[T], b: int, i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a: del self.a[b]
        return ans
    def remove(self, x: T) -> bool:
        if self.size == 0: return False
        a, b, i = self._position(x)
        if i == len(a) or a[i] != x: return False
        self._pop(a, b, i)
        return True
    def lt(self, x: T) -> Optional[T]:
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]
    def le(self, x: T) -> Optional[T]:
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]
    def gt(self, x: T) -> Optional[T]:
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]
    def ge(self, x: T) -> Optional[T]:
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    def __getitem__(self, i: int) -> T:
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0: return a[i]
        else:
            for a in self.a:
                if i < len(a): return a[i]
                i -= len(a)
        raise IndexError
    def pop(self, i: int = -1) -> T:
        if i < 0:
            for b, a in enumerate(reversed(self.a)):
                i += len(a)
                if i >= 0: return self._pop(a, ~b, i)
        else:
            for b, a in enumerate(self.a):
                if i < len(a): return self._pop(a, b, i)
                i -= len(a)
        raise IndexError
    def index(self, x: T) -> int:
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans
    def index_right(self, x: T) -> int:
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans
    def find_closest(self, k: T) -> Optional[T]:
        if self.size == 0:
            return None
        ltk = self.le(k)
        gtk = self.ge(k)
        if ltk is None:
            return gtk
        if gtk is None:
            return ltk
        if abs(k-ltk)<=abs(k-gtk):
            return ltk
        else:
            return gtk
sys.setrecursionlimit(10**5)
def POW(base, exp, mod):
    result = 1
    base = base % mod  # Handle case when base is larger than mod
    while exp > 0:
        if exp % 2 == 1:  # If exp is odd, multiply base with result
            result = (result * base) % mod
        exp = exp // 2    # Divide exp by 2
        base = (base * base) % mod  # Square the base
    return result
def IL():
  return [int(i) for i in input().split()]
def CL():
  return [i for i in input().split()]
def I():
  return input()
def inti():
  return int(input())
def db(x):
  return print(x)
def dbl(x):
  return print(*x)
def dbm(x):
  for i in x:
    print(i)
def sq(x):
  return x==int(x**0.5)**2
def prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if (n & 1) == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

char="abcdefghijklmnopqrstuvwxyz"
mod=pow(10,9)+7
for _ in range(int(input())):
  n,k=IL()
  arr=IL()
  
