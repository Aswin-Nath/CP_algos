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
class TrieNode:
    def __init__(self,val=-1):
        self.child={}
        self.val=val
class Trie:
    def __init__(self):
        self.root=TrieNode()
    def insert(self,word):
        root=self.root
        for i in word:
            if i not in root.child:
                root.child[i]=TrieNode(i)
            root=root.child[i]
    def prefix(self,word):
        i=0
        root=self.root
        while i<len(word):
            if not root:
                return False
            if word[i] not in root.child:
                return False
            root=root.child[i]
            i+=1
        return True
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

def miller_is_prime(n):
    """
        Miller-Rabin test - O(7 * log2n)
        Has 100% success rate for numbers less than 3e+9
        use it in case of TC problem
    """
    if n < 5 or n & 1 == 0 or n % 3 == 0:
        return 2 <= n <= 3
    s = ((n - 1) & (1 - n)).bit_length() - 1
    d = n >> s
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        p = pow(a, d, n)
        if p == 1 or p == n - 1 or a % n == 0:
            continue
        for _ in range(s):
            p = (p * p) % n
            if p == n - 1:
                break
        else:
            return False
    return True

import random
RANDOM = random.randrange(1,2**62)
def w(x):
    return x ^ RANDOM
class Xdict:
    def __init__(self,L=[],flag=1):
        self.d = {}
        self.flag = flag
        for j in L:self[j]+=1
    def __setitem__(self,key,value):
        self.d[w(key)] = value
    def __getitem__(self,key):
        return self.d.get(w(key),0) if self.flag else self.d[w(key)]
    def keys(self):
        return (w(i) for i in self.d)
    def values(self):
        return (self.d[i] for i in self.d)
    def items(self):
        return ((w(i),self.d[i]) for i in self.d)
    def __repr__(self):
        return '{'+','.join([str(w(i))+':'+str(self.d[i]) for i in self.d])+'}'
    def __delitem__(self,val):
        del self.d[w(val)]
    def get(self,key,other):
        return self.d.get(w(key),other)
    def __contains__(self,key):
        return w(key) in self.d
    def __len__(self):
        return len(self.d)
    def clear(self):
        self.d.clear()
    def __iter__(self):
        return iter(self.keys())

def sieve(n):
    primes = []
    isp = [1] * (n+1)
    isp[0] = isp[1] = 0
    for i in range(2,n+1):
        if isp[i]:
            primes.append(i)
            for j in range(i*i,n+1,i):
                isp[j] = 0
    return primes

def all_fact(n):
    """
    returns a sorted list of all distinct factors of n in root n
    """
    small, large = [], []
    for i in range(1, int(n**0.5) + 1, 2 if n & 1 else 1):
        if not n % i:
            small.append(i)
            large.append(n // i)
    if small[-1] == large[-1]:
        large.pop()
    large.reverse()
    small.extend(large)
    return small

class Xset:
    def __init__(self,L=[]):
        self.s = set()
        for j in L:self.add(j)
    def add(self,key):
        self.s.add(w(key))
    def keys(self):
        return (w(i) for i in self.s)
    def __repr__(self):
        return '{'+','.join([str(w(i)) for i in self.s])+'}'
    def remove(self,val):
        self.s.remove(w(val))
    def __contains__(self,key):
        return w(key) in self.s
    def __len__(self):
        return len(self.s)
    def clear(self):
        self.s.clear()
    def __iter__(self):
        return iter(self.keys())
def get_matrix_input(N,typ):
  G=[]
  for i in range(N):
    if typ=="str":
      get=list(input())
    else:
      get=IL()
    G.append(get)
  return G
char="abcdefghijklmnopqrstuvwxyz"
mod=pow(10,9)+7
calc = False
if calc:
    def sieve_unique(N):
        mini = [i for i in range(N)]
        for i in range(2,N):
            if mini[i]==i:
                for j in range(2*i,N,i):
                    mini[j] = i
        return mini

    MAX_N = 10**6+1
    Lmini = sieve_unique(MAX_N)

    def prime_factors(k,typ=0):
        """
            When the numbers are large this is the best method to get
            unique prime factors, precompute n log log n , then each query is log n
        """
        if typ==0:
            ans = Counter()
        elif typ==1:
            ans = set()
        else:
            ans = []
        while k!=1:
            if typ==0:
                ans[Lmini[k]] += 1
            elif typ==1:
                ans.add(Lmini[k])
            else:
                ans.append(Lmini[k])
            k //= Lmini[k]
        return ans

    def all_factors(x):
        # returns all factors of x in log x + d
        L = list(prime_factors(x).items())
        st = [1]
        for i in range(len(L)):
            for j in range(len(st)-1,-1,-1):
                k = L[i][0]
                for l in range(L[i][1]):
                    st.append(st[j]*k)
                    k *= L[i][0]
        return st
from types import GeneratorType
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

t=int(input())
# t=1
for _ in range(t):
  n=int(input())
  arr=IL()
  
