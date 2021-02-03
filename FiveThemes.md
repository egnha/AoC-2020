# Five Themes from the [`Advent of Code 2020`](https://adventofcode.com/2020) 🎄

***Eugene Ha***
<br>
([License](#license))

This is a commentary on nine days from the [`Advent of Code 2020`](https://adventofcode.com/2020), categorized by theme.

  * [Theme 1](#theme1). Counting with linear algebra (Day [7](#day7))
  * [Theme 2](#theme2). Array methods for cellular automata (Days [11](#day11) and [17](#day17)) 
  * [Theme 3](#theme3). Geometry with complex numbers (Days [12](#day12) and [24](#day24))
  * [Theme 4](#theme4). Poor man’s interpreter (Days [14](#day14) and [18](#day18))
  * [Theme 5](#theme5). Resolving ambiguous relations with matrices (Days [16](#day16) and [21](#day21))
  
Solutions to the remaining sixteen days can be found in the [solutions folder](https://github.com/egnha/AoC-2020/tree/main/solutions). They are comparatively routine (with the exception of [Day 20](https://github.com/egnha/AoC-2020/blob/main/solutions/20.ipynb), which is rather delicate). I’ve omitted them here because five is an auspicious number, and frankly, I lacked the imagination or inspiration (not to mention the energy) to say anything noteworthy about them.

All solutions are in Python, with liberal use of [NumPy](https://numpy.org) and [array-oriented methods](https://en.wikipedia.org/wiki/Array_programming). I favour a functional style, but write procedural or imperative code when that makes more sense. There is scarcely any object orientation, except as a means of overriding ordinary Python semantics ([Theme 4](#theme4)).

Hearty thanks to [Eric Wastl](http://www.was.tl) for creating such a fun and educational diversion!

#### Dependencies

[Python 3.8](https://docs.python.org/3.8/) (or higher) and [NumPy](https://numpy.org) are required. [SciPy](https://www.scipy.org/scipylib/) and [Numba](https://numba.pydata.org) are each used once for speed-ups ([Day 7](#day7), resp. [Day 11](#day11)).


```python
import numpy       as np
import collections as co
import itertools   as it
import functools   as ft
import operator    as op
import re
```

#### Type annotations

Function-type annotations are used as just that — *annotations*. I’ve applied them in places where they might aid comprehension, but stopped short of verifying their consistency with a static type checker.


```python
from typing import Callable, Iterator, List, Tuple, Dict, NewType

Vector = NewType("Vector", np.ndarray)  # implicitly 1D
Matrix = NewType("Matrix", np.ndarray)  # implicitly 2D
```

<a id='theme1'></a>
## Theme 1. Counting with linear algebra (Day 7)

<a id='day7'></a>
### [Day 7](https://adventofcode.com/2020/day/7): *Handy Haversacks*

The key observation is that **bag-containment counts are linearly related.**

For example, consider the containment rule `blue bags contain 4 green bags, 2 red bag, 5 yellow bags`. We can transcribe the corresponding condition for the containment *count* — the number of bags contained by a blue bag — as a *purely symbolic* equation:

```
N_blue = 4⋅(𝟙_green + N_green) + 2⋅(𝟙_red + N_red) + 5⋅(𝟙_yellow + N_yellow)
```

The symbolic summand `4⋅(𝟙_green + N_green)` accounts for 4 green bags plus 4 times the number of bags contained each green bag; likewise for the other summands. There are no linear relations among the symbols `𝟙_green`, `𝟙_red`, … (whatever they happen to “be”).

All this suggests the following semantics: *The symbols `N_blue`, `N_green`, … are vectors in a vector space spanned by the basis `𝟙₀`, `𝟙₁`, …, `𝟙ᵣ` (of formal symbols), where `0`, `1`, …, `r` is an (arbitrary) enumeration of the variously coloured bags*.

Let `C` be the square matrix whose entry `C[i, j]` is the number of times bag `i` contains bag `j`. Then the system of containment-count equations is the matrix equation

```
N = C + C⋅N
```

The problem is to determine the matrix `N`. The number of bags contained in bag `i` is then the **sum of the entries in the `i`-th row of `N`**.

Now to the code. Standard manipulations determine the matrix `C` from the puzzle data.


```python
rule = re.compile(r"(.*) bags contain (.*)\.")
bags = re.compile(r"(\d+) (.*?) bag")

with open("data/07.txt", "r") as f:
    rules = rule.findall(f.read())
    enum = {bag: i for i, (bag, _) in enumerate(rules)}
    C = np.zeros((len(enum), len(enum)), dtype=int)
    for bag, innerbags in rules:
        for n, innerbag in bags.findall(innerbags):
            C[enum[bag], enum[innerbag]] = int(n)
```

Basic algebra yields the solution of the equation `N = C + C⋅N` as an (infinite) geometric series:

```
N = C + C² + C³ + …
```

This is in fact a *finite* sum if sufficiently high powers of `C` are zero (which is to say that `C` is “nilpotent”). Assuming this, the following function computes the series:


```python
from scipy.sparse import csr_matrix

def geomseries(x: Matrix) -> Matrix:
    """Geometric series starting from x (presuming it's nilpotent and sparse)."""
    x = csr_matrix(x)
    pows = it.accumulate(it.repeat(x), op.matmul, initial=x)
    return np.sum(list(it.takewhile(csr_matrix.count_nonzero, pows)), axis=0)
```

Our matrix `C` is indeed nilpotent (why?), so we can now solve **Part 2**. The number of bags inside a shiny gold bag is **`13264`**.


```python
N = geomseries(C)
shinygold = enum["shiny gold"]

assert 13264 == np.sum(N[shinygold])
```

**Part 1** concerns a *boolean predicate* — which bags contain at least one shiny gold bag? — for the *inverse* containment relationship. This amounts to transposing the array `C` and casting it to type `bool`. We find that the number of bags that can eventually contain at least one shiny gold bag is **`222`**.


```python
assert 222 == np.sum(geomseries(C.T.astype(bool))[shinygold])
```

#### An elementary approach via recursion

Here’s an elementary approach without matrix algebra or geometric series. Say we’re interested in bag `i`. As before, consider its bag-containment-count equation:

```
Nᵢ = Cᵢ₀⋅(N₀ + 𝟙₀) + Cᵢ₁⋅(N₁ + 𝟙₁) + … + Cᵢᵣ⋅(Nᵣ + 𝟙ᵣ)
```

Now substitute into this the corresponding equations for `N₀`, `N₁`, …, and continue making such substitutions until only a linear combination of the basis vectors `𝟙₀`, `𝟙₁`, …, remains:

```
Nᵢ = … = aᵢ₀⋅𝟙₀ + aᵢ₁⋅𝟙₁ + … + aᵢᵣ⋅𝟙ᵣ
```

Here’s a function that performs recursive substitution:


```python
arithmetic = {np.dtype(bool): (op.or_, op.and_), # Part 1
              np.dtype(int):  (op.add, op.mul)}  # Part 2

def substitute(C: Matrix, i: int) -> Vector:
    """Recursively substitute bag-containment-count equations."""
    add, mul = arithmetic[C.dtype]
    sum = lambda xs: np.sum(list(xs), axis=0, dtype=C.dtype)
    def subst(N):
        if (N == 0).all():
            return N
        return add(N, sum(mul(n, subst(c)) for n, c in zip(N, C) if n))
    return subst(C[i])
```

Parts 1 and 2 now proceed as before.


```python
assert 222 == np.sum(substitute(C.T.astype(bool), shinygold))  # Part 1
assert 13264 == np.sum(substitute(C, shinygold))               # Part 2
```

<a id='theme2'></a>
## Theme 2. Array methods for cellular automata (Days 11 and 17)

For Days 11 and 17, we adapt the array methods in [John M. Scholes](https://en.wikipedia.org/wiki/John_M._Scholes)’s astounding demo of the [Conway’s Game of Life in APL](https://youtu.be/a9xAKttWgP4).

<a id='day17'></a>
### [Day 17](https://adventofcode.com/2020/day/17): *Conway Cubes*

Compose the following functions to **simulate** a 6-cycle boot process:

```
      nest        cycle         cycle      cycle
cubes ----> CUBES -----> CUBES1 -----> … -----> CUBES6
```

Here `cubes` is the initial grid of cubes as a 2-dimensional boolean array (according to the predicate, “Is a cube active?”); `nest` takes an array and nests it in a higher-dimensional array, by repeated boxing; `cycle` is an application of a cycle of the boot process (a “tick” of a cellular automaton).

The beauty of using arrays is that it enables you to **treat the entire collection of cubes as an atomic unit**. No delicate index fiddling is necessary to figure out how the cubes affect each other.

To (repeatedly) box a NumPy array, apply the `ndarray.reshape()` method. For example, to box a 3-dimensional array `x` of shape `(p, q, r)` twice to get a 5-dimensional array, do `x.reshape(1, 1, p, q, r)`.


```python
def nest(x: np.ndarray, dim: int) -> np.ndarray:
    boxes = np.ones(dim - len(x.shape), dtype=int)
    return x.reshape(*boxes, *x.shape)
```

Each **cycle** of the boot process is executed by computing the **density** of cubes — a count of active neighboring cubes, *including* the cube in question — and then activating or deactivating cubes according to the activation predicate, “Is the density 3, or is the density 4 and the cube is active?” This is essentially [Conway’s Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life). To allow cube activation to propagate outward, we **extend** the (nested) array of cubes by appending a border of `False`’es in each dimension. Because extension intervenes each time a cycle or density is computed, it is more natural to think of it as a *functional* transformation (i.e., function decorator) rather than as an array transformation.


```python
def boot(dim: int):
    """A boot cycle in dim dimensions."""
    views = list(it.product(shifts, repeat=dim))
    @extend
    def density(cubes: np.ndarray) -> np.ndarray:  # After John M. Scholes
        return np.sum([cubes[v] for v in views], axis=0)
    @extend
    def cycle(cubes: np.ndarray) -> np.ndarray:
        d = density(cubes)
        return (d == 3) | ((d == 4) & cubes)
    return cycle

shifts = np.s_[:-2], np.s_[1:-1], np.s_[2:]

def extend(f):
    """Extend an array by 0 before applying f."""
    return lambda x: f(np.pad(x, 1))
```

The **simulation** then proceeds by nesting the cubes in a higher-dimensional space, then cycling.


```python
def simulate(cubes: np.ndarray, cycles: int, dim: int) -> np.ndarray:
    """Simulate n cycles of the boot process in dimension dim."""
    cycle = boot(dim)
    return ft.reduce(lambda c, _: cycle(c), range(cycles), nest(cubes, dim))
```

For **Part 1**, we find that the number of cubes left in the active state after the sixth cycle is **`426`**.


```python
with open("data/17.txt", "r") as f:
    lines = [l.strip() for l in f.readlines()]
    cubes = np.array(list(map(list, lines))) == "#"

assert 426 == np.sum(simulate(cubes, cycles=6, dim=3))
```

For **Part 2**, the ambient dimension is 4, and the numbers of cubes left in the active state after the sixth cycle is now **`1892`**.


```python
assert 1892 == np.sum(simulate(cubes, cycles=6, dim=4))
```

<a id='day11'></a>
### [Day 11](https://adventofcode.com/2020/day/11): *Seating System*

If occupying a seat in Day 11 corresponds to activating a cube in Day 17, then Part 1 is identical to Day 17, with two minor differences:

  1. the rule of seat occupation differs from the rule of cube activation;
  2. reseating runs until it stabilizes, instead of running a fixed number of times.


```python
Predicate = NewType("Predicate", Matrix)  # dtype: bool
Count = NewType("Count", Matrix)          # dtype: int
Density = Callable[[Predicate], Count]

def reseating(seats: Predicate, density: Density, threshold: int):
    """Reseat according to occupation density."""
    def reseat(occupied):
        d = density(occupied)
        return (seats & (d == 0)) | (occupied & (d <= threshold))
    return reseat

@extend  # cf. Day 17
def neighbors(occupied: Predicate) -> Count:
    """Count the number of occupied neighboring seats."""
    return np.sum([occupied[s] for s in shifts2], axis=0)

shifts2 = list(it.product(shifts, repeat=2))  # cf. Day 17

def fixedpoint(f, x):
    while ((fx := f(x)) != x).any():
        x = fx
    return fx
```

For **Part 1**, we find that **`2344`** seats are occupied once reseating stabilizes.


```python
with open("data/11.txt", "r") as f:
    seatingplan = np.array([list(l.strip()) for l in f.readlines()])
    seats = seatingplan == "L"

reseat = reseating(seats, density=neighbors, threshold=4)

assert 2344 == np.sum(fixedpoint(reseat, seats))
```

For **Part 2**, both the occupation density and its threshold increase. The threshold becomes 5 seats, and the occupation density now counts any occupied seat that is in direct line of sight. The latter is computed by a closure `lineofsight()`, which depends on the seating layout. An iterable `visibility` collects the functions that count the occupied seats in each of the eight lines of sight. Creating `visibility` is the bulk of the effort for Part 2.


```python
UNOCCUPIED = -np.inf

def visible(seats: Predicate) -> Density:
    """Count occupied seats that are in direct line of sight."""
    def lineofsight(occupied: Predicate) -> Count:
        occ = occupied.astype(float)
        occ[seats & ~occupied] = UNOCCUPIED
        return np.sum([vis(occ) & seats for vis in visibility], axis=0)
    return lineofsight
```

Here `UNOCCUPIED` is a sentinel value for unoccupied seats. The reason this is chosen to be minus infinity will be clear shortly.

Here’s a 3-step method for determining the **occupied seats for which an occupied seat is visible above**:

  1. Cast the boolean matrix of occupied seats to a floating-point matrix, `occ`; moreover, assign unoccupied seats the value `UNOCCUPIED`. (Thus `occ` entries are either `0`, `1`, or `UNOCCUPIED`.)
  2. (**Main step**) Determine the points for which an occupied seat is visible above, `visibleabove(occ)`.
  3. Among these points, restrict to the subset of seats, `visibleabove(occ) & seats`.

  
```
    occ                visibleabove(occ)

. L # . . .               . . . . . .                 . . . . . .
. . # L # .               . . ^ . . .                 . . * . . .
. . . . L L    look up    . . ^ . ^ .    _ & seats    . . . . * .
# . . . . .   -------->   . . ^ . . .   ---------->   . . . . . .
. . L # . .               ^ . ^ . . .                 . . . . . .
. . . L L .               ^ . . ^ . .                 . . . * . .

                      (^: # visible above)
```

Step 2 coincides with taking a **“rectified” cumulative sum** down the columns of `occ` — apply the **rectified linear unit** `max(0, x)` as the intermediate sums are accumulated. Because unoccupied seats have value `-∞`, they block the visibility of any occupied seats above them.


```python
import numba as nb

Occ = NewType("Occ", Matrix)  # dtype: float

def visibleabove(occ: Occ) -> Predicate:
    """Is an occupied seat visible above?"""
    occ = np.pad(occ[:-1], ((1, 0), (0, 0)))  # initial value 0
    return reluadd.accumulate(occ).astype(bool)

@nb.vectorize([nb.float64(nb.float64, nb.float64)], nopython=True)
def reluadd(x, y):
    return np.maximum(0, x + y)
```

[Numba](https://numba.pydata.org/) is used here to compile `reluadd()` to a so-called “ufunc,” in order to get the `reluadd.accumulate()` method for computing the rectified cumulative sum (with initial value `0`). The function `itertools.accumulate()` would also work, but the result would be verbose and considerably slower.

Now change perspective to determine the occupied seats in the other directions. For non-diagonal lines of sight, this amounts to some mix of transposing and reversing (realized as functional transformations).


```python
def rev(f: Callable[[Matrix], Matrix]):
    return lambda x: f(x[::-1])[::-1]

def transpose(f: Callable[[Matrix], Matrix]):
    return lambda x: f(x.T).T

visiblebelow = rev(visibleabove)
visibleleft = transpose(visibleabove)
visibleright = transpose(visiblebelow)
```

Handling diagonal lines of sight is more involved because standard array idioms only apply to rows and columns. A simple way to resolve this mismatch is to **shear** the matrix horizontally so that diagonals are tilted into columns. For example, anti-diagonals can be tilted vertically by shearing the bottom to the right:

```
 (x)                 (diag)
 
0 1 2             0 1 2 . . . .
1 2 3    shear    . 1 2 3 . . .
2 3 4   ------>   . . 2 3 4 . .
3 4 5             . . . 3 4 5 .
4 5 6             . . . . 4 5 6
```

This picture translates to the following code:


```python
import numpy.ma as ma

def shear(x: Matrix, void=UNOCCUPIED) -> ma.MaskedArray:
    """Shear the bottom of a matrix to the right."""
    m, n = x.shape
    i, j = np.indices((m, n))
    j = j + np.arange(m)[:, np.newaxis]
    diag = np.full((m, m + n - 1), void)
    mask = np.ones(diag.shape, dtype=bool)
    diag[i, j] = x
    mask[i, j] = False
    return ma.array(diag, mask=mask)
```

(Packaging the result of `shear()` as a masked array is a mere technical convenience. It enables array operations to treat the data and its mask as a unit.)

With shearing as intermediary, methods for determining occupied seats in *vertical* lines of sight can be used to determine occupied seats in *diagonal* lines of sight.


```python
def visiblealong(shear: Callable[[Matrix], ma.MaskedArray]):
    def visibility(vis: Callable[[Occ], Predicate]):
        """Is a point visible from below an occupied seat on axis?"""
        def visible(occ: Occ) -> Predicate:
            d = shear(occ)
            return vis(d.data)[~d.mask].reshape(occ.shape)
        return visible
    return visibility

diagonally = visiblealong(rev(shear))
antidiagonally = visiblealong(shear)
```

Lastly, we must regard occupied seats as being visible to themselves. (Alternatively, take this into account by modifying the occupation rule in `reseating()`.)


```python
def visibleitself(occ: Occ) -> Predicate:
    return occ == 1  # occupied seat is visible to itself
```

By collecting all the lines of sight, we now get the iterable `visibility`.


```python
visibility = [visibleitself,
              visibleabove,
              visiblebelow,
              visibleleft,
              visibleright,
              diagonally(visibleabove),
              diagonally(visiblebelow),
              antidiagonally(visiblebelow),
              antidiagonally(visibleabove)]
```

For **Part 2**, we find that **`2076`** seats are occupied once reseating according line-of-sight visibility stabilizes.


```python
reseat2 = reseating(seats, density=visible(seats), threshold=5)

assert 2076 == np.sum(fixedpoint(reseat2, seats))
```

<a id='theme3'></a>
## Theme 3. Geometry with complex numbers (Days 12 and 24)

The key idea for days 12 and 24 is a lesson from high-school math: **identify points in a plane with complex numbers to convert geometry to arithmetic**. Translation of points corresponds to addition. Rotation corresponds to multiplication by a complex number of absolute value 1.

<a id='day12'></a>
### [Day 12](https://adventofcode.com/2020/day/12): *Rain Risk*

Regard the ship’s position and direction as a pair of complex numbers, `pd = (p, d)`. Then turns and forward movement are linear transformations of the 2-vector `pd`. But movement in a compass direction is a translation by a constant, and is therefore a *non*-linear transformation of `pd`.

There’s a trick for treating linear transformations and translations on an equal footing, familiar from computer graphics and group-representation theory: **represent the 2-vector `(p, d)` as a 3-vector `(p, d, 1)`**
. Then both linear transformations and translations of *embedded* 2-vectors are performed by linear transformations of the ambient 3-dimensional space. Specifically, moving in a compass direction `x`, moving forward by `y`,  turning by `d` degrees are each performed by multiplying the (row) vector `(p, d, 1)` on the *right* by one of the following matrices:

```
Move     Forward    Turn

1 0 0     1 0 0     1 0 0
0 1 0     y 1 0     0 t 0  (t = exp(√-1⋅π⋅d/180))
x 0 1     0 0 1     0 0 1
```

To generate navigation matrices, we can use a closure; the choice of `move` (a matrix-valued function) distinguishes Part 1 from Part 2.


```python
Move = Callable[[complex], Matrix]

def navigation(move: Move) -> Callable[[str, int], Matrix]:
    """Generate navigation matrices."""
    nav = dict(F=forward,
               L=lambda v: turn(L**(v//90)),
               R=lambda v: turn(R**(v//90)),
               N=lambda v: move(N*v),
               S=lambda v: move(S*v),
               E=lambda v: move(E*v),
               W=lambda v: move(W*v))
    return lambda action, value: nav[action](value)

E, N, W, S = +1, +1J, -1, -1J
L, R = +1J, -1J

def move(x: complex):
    return np.array([[1, 0, 0], [0, 1, 0], [x, 0, 1]])

def forward(y: int):
    return np.array([[1, 0, 0], [y, 1, 0], [0, 0, 1]])

def turn(t: complex):
    return np.diag([1, t, 1])
```

To navigate the ship to a new position, start from the initial `(position, direction, 1)` 3-vector, and successively right-multiply by matrices corresponding to the navigation instructions (the puzzle input). Doing repeated vector-matrix multiplications is more efficient that doing a big matrix multiplication upfront.


```python
def navigator(instructions: List[Tuple[str, int]]):
    def navigate(move: Move, facing: complex) -> complex:
        """Navigate the ship to a new position."""
        nav = navigation(move)
        start = (0, facing, 1)
        end, *_ = ft.reduce(op.matmul, it.starmap(nav, instructions), start)
        return end
    return navigate

with open("data/12.txt", "r") as f:
    instructions = [(inst[0], int(inst[1:])) for inst in f.readlines()]
    navigate = navigator(instructions)
```

Now we can dispatch **Part 1**. After navigating the ship, which is initially facing east, the Manhattan distance to the starting position is **`1457`**.


```python
def manhattan(z: complex):
    return abs(z.real) + abs(z.imag)

assert 1457 == manhattan(navigate(move, facing=E))
```

For **Part 2**, the effect of the waypoint is to flip the roles of position and direction when moving in a compass direction, which amounts to flipping the function `move()`. The Manhattan distance to the starting position is now **`106860`**, when the ship’s initial direction qua “waypoint” is 10 units east and 1 unit north.


```python
F = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

def flip(move: Move) -> Move:
    return lambda x: F @ move(x) @ F

assert 106860 == manhattan(navigate(flip(move), facing=10*E + N))
```

The transformation `flip()` elucidates the relationship between the two parts, though it’d be more economical to dispense with the factor `F`, by substituting `flip(move)` with an explicit matrix-function of `x`.

<a id='day24'></a>
### [Day 24](https://adventofcode.com/2020/day/24): *Lobby Layout*

Imagine for a moment that the tiles are situated on a *regular* hexagonal grid in the complex numbers, centered at `0` (the reference tile). Then the **endpoint of a path in this grid**, starting from `0`, is a **sum of sixth roots of unity**, where a *sixth root of unity* is a complex number `z` such that `z⁶ = 1`. If `z` is the (primitive) sixth root of unity `exp(√-1⋅π/3)`, then the directions *e*, *ne*, *nw*, *w*, *sw*, *se* correspond to `1`, `z`, `z - 1`, `-1`, `-z`, `-z + 1`, respectively. Under this correspondence, the endpoint of the path *wwswenenw* is the complex number `z - 2`:

```
 w      w      sw    e   ne   nw
(-1) + (-1) + (-z) + 1 + z + (z - 1) = z - 2
```

In fact, whether the lattice is regular or skewed is irrelevant, because we only want to *tally* the times a grid point is reached by a path. To this end, only the *linear independence* of `1` and `z` over the real numbers matters. We might as well choose `z = √-1`. (From a planar perspective, the hexagonal grid becomes skewed.)


```python
neighbors = dict(e=1+0J, ne=1J, nw=-1+1J, w=-1+0J, sw=-1J, se=1-1J)
```

A grid point (i.e., tile) is “blackened” (flipped to black) if the number paths reaching it is odd.


```python
def blacken(paths: List[List[str]]) -> Vector:
    visits = co.Counter(sum(map(neighbors.__getitem__, p)) for p in paths)
    return np.array([t for t, v in visits.items() if v % 2], dtype=complex)
```

Now we can answer **Part 1**. After traversing all paths (flipping all tiles) the number of grid points that are blackened is **`351`**.


```python
with open("data/24.txt", "r") as f:
    paths = [re.findall("e|ne|nw|w|sw|se", line.strip()) for line in f.readlines()]

black = blacken(paths)

assert 351 == len(black)
```

For **Part 2**, the blackening (flipping) rule depends on neighboring tiles, rather than paths to a tile. But the structure of the computation itself is unchanged. We find that **`3869`** tiles are blackened after 100 days.


```python
shift = np.fromiter(neighbors.values(), dtype=complex).reshape(-1, 1)

def flip(black: np.ndarray, _) -> Vector:
    """Flip to black according to the color of neighboring tiles."""
    nbhs = co.Counter((black + shift).flat)
    return np.concatenate(([b for b, n in nbhs.items() if n == 2],
                           [b for b in black if nbhs.get(b) == 1]))

assert 3869 == len(ft.reduce(flip, range(100), black))
```

<a id='theme4'></a>
## Theme 4. Poor man’s interpreter (Days 14 and 18)

Days 14 and 18 amount to building a **poor man’s intepreter**: we stipulate semantics, Python does the parsing and evaluation.

<a id='day14'></a>
### [Day 14](https://adventofcode.com/2020/day/14): *Docking Data*

Observe that the **puzzle input is basically Python code** — just run it.

In other words, presuming `code` is the puzzle input as a (big) string, we want to do something like this to get the sum of memory values:

```python
exec(code)
sum(mem)
```

The fact that the input is not 100% syntactically valid Python is a minor obstacle. For example, the right-hand side of a line like

```
mask = 01X11X10X10110110X111X11010X1X101010
```

is syntactically invalid. But we can fix that by treating it as a string.

Thus one strategy for Day 14 is an **ad hoc tweak of the Python interpreter**, in two steps:

  1. Do a light code transformation to get syntactically valid Python.
  2. Reinterpret the assignment operator so that bitmasking occurs.

The **first step** is performed by a simple “stringification” of mask values.


```python
with open("data/14.txt", "r") as f:
    code = re.sub(r"mask = (.*)\n", r"mem.mask = '\1'\n", f.read())
```

The additional code transformation

```
mask = …  ⤳  mem.mask = …
```

enables the **second step**: reinterpreting mask assignment and memory value assignment so that bitmasking intervenes. Since we have lines like `mem.mask = …` and `mem[…] = …` in the code, it is natural to endow `mem` with these capabilities by making it an instance of an ad hoc class. One possibility for such a class is as follows:


```python
class Memory(Dict[int, int]):
    """Memory as a dictionary of address-value pairs."""
    
    @classmethod
    def write(cls, registers, code: str):
        """Write to memory by running code."""
        cls._registers = registers
        mem = cls()  # Must be named "mem"
        exec(code)
        return mem

    def _registers(self, addr: int, val: int) -> Iterator[Tuple[int, int]]:
        """Generate address-value pairs to write."""
    
    def _setmask(self, val: str):
        self._mask = self._int("1" if x == "X" else "0" for x in val)
        self._places = [i for i, x in enumerate(reversed(val)) if x == "X"]
        self._overwrite = self._int("0" if x == "X" else x for x in val)

    @staticmethod
    def _int(x: str) -> int:
        return int("".join(x), 2)
    
    mask = property(fset=_setmask)

    def __setitem__(self, addr: int, val: int):
        """Write values to memory."""
        for a, v in self._registers(addr, val):
            super().__setitem__(a, v)
        
    def __iter__(self) -> Iterator[int]:
        return iter(super().values())
```

The method `Memory._registers()` is *dynamically* assigned by the constructor `Memory.write()`. For Part 1, it maskes memory values; for Part 2, it masks many memory addresses.


```python
def registers1(self, addr: int, val: int) -> Iterator[Tuple[int, int]]:
    """Apply bitmask to memory values."""
    yield addr, (val & self._mask) | self._overwrite

def registers2(self, addr: int, val: int) -> Iterator[Tuple[int, int]]:
    """Decode memory addresses."""
    addr = (addr | self._overwrite) & (BITS36 ^ self._mask)
    for bits in it.product((0, 1), repeat=len(self._places)):
        yield addr | sum(map(op.lshift, bits, self._places)), val

BITS36 = 2**36 - 1
```

We can now solve both parts by **literal interpretation**. For **Part 1**, the sum of all values left in memory after the code runs is **`7817357407588`**.


```python
assert 7817357407588 == sum(Memory.write(registers1, code))
```

For **Part 2**, the sum of all values left in memory after the code runs is **`4335927555692`**.


```python
assert 4335927555692 == sum(Memory.write(registers2, code))
```

<a id='day18'></a>
### [Day 18](https://adventofcode.com/2020/day/18): *Operation Order*

The problem is to **reinterpret the syntax of operators** in an arithmetic expression. For each operator, this means independently settings its **precedence** and **meaning**. In Python, the precedence of an operator is fixed, but its meaning is changeable.

We can therefore proceed in three steps:

  1. Substitute each operator with a Python operator of the appropriate precedence.
  2. For some ad hoc subclass of `int`, reinterpret each such operator.
  3. Evaluate the resulting expression.

The first two steps amount to taking a (newline-separated) *sequence* of arithmetic expressions like

```
(8 + 5) + 1 + (2 * 3 * 4)
7 + 8 * (5 + (7 + 3 * 3))
…
```

and combining them into a *single* expression (without newlines):

```
(Weird(8) / Weird(5)) / Weird(1) / (Weird(2) * Weird(3) * Weird(4)) |
Weird(7) / Weird(8) * (Weird(5) / (Weird(7) / Weird(3) * Weird(3))) |
…
```

Here `Weird` is some ad hoc subclass of `int` that implements addition via `/`, in order that addition and multiplication have equal precedence (as dictated by Part 1). Each newline, which implicitly executes addition, is substituted by an explicit `|` operator of relatively low precedence in Python. The resulting expression is huge, but still manageable for Python’s parser.

All this leads directly to the following implementation of the class `Weird` and the “weirdification” of expressions:


```python
class Weird(int):
    """Integers with weird arithmetic syntax."""
    
    add = lambda self, other: Weird(super().__add__(other))
    mul = lambda self, other: Weird(super().__mul__(other))
    
    __or__      = add  # `|`
    __truediv__ = add  # `/`
    __add__     = mul  # `+`
    __mul__     = mul  # `*`

def weird(exprs):
    """Make a bunch of arithmetic expressions weird."""
    wexprs = exprs.strip().replace("\n", "|").replace("+", "/")
    wexprs = re.sub(r"(\d+)", r"Weird(\1)", wexprs)
    return wexprs
```

If `wexprs` is the puzzle input made weird, then the value of its interpretation is the answer to **Part 1**, namely **`5374004645253`**.


```python
with open("data/18.txt", "r") as f:
    wexprs = weird(f.read())

assert 5374004645253 == eval(wexprs)
```

When `*` is substituted everywhere by `+` (which then does multiplication at the precedence of *Python* addition), we get the answer to **Part 2**, namely **`88782789402798`**.


```python
assert 88782789402798 == eval(wexprs.replace("*", "+"))
```

<a id='theme5'></a>
## Theme 5. Resolving ambiguous relations with matrices (Days 16 and 21)

Day 21 and Part 2 of Day 16 boil down to **resolving ambiguous relations**: from a given relation (subset) `R ⊆ A × B` between two finite sets `A` and `B`, **find a one-to-one function** `A → B` (presuming `B` has no fewer elements than `A`, of course).

Upon identifying `A` and `B` with an enumeration of their elements, we can translate the resolution of ambiguity into a problem about the **incidence matrix** of `R`, which is the boolean matrix `p` whose entry `p[a, b]` is true if and only if `(a, b) ∈ R`. Resolving the ambiguity of `R` then amounts to finding an array `ch` (indexed by `A`) of *distinct* elements of `B` such that `p[a, ch[a]]` is true for every `a ∈ A`. Call an array with this property a **choice** for `p` (or `R`).

For example, if `A` has 5 elements and `B` has 7 elements, then the incidence matrix of `R` is a 5×7 matrix `p`, and a choice for `p` is then a selection of 5 true entries of `p` in distinct rows and columns (indicated here by `*`):

```
0 0 1 0 0 0 0              0 0 * 0 0 0 0
1 0 1 1 0 0 0    choice    1 0 1 * 0 0 0
0 0 0 0 0 1 0   ------->   0 0 0 0 0 * 0
1 0 1 1 1 0 1              * 0 1 1 1 0 1
0 0 1 0 1 1 0              0 0 1 0 * 1 0
```

One procedure that attempts (but sometimes fails) to find a choice for `p` goes like this: find the row indices `a` for which the choice of a column index `ch[a]` is unambiguous, then excise those rows and columns from `p`, and recurse. A column index `b` is **unambiguous** when its column `p[:, b]` has a unique true value *among the rows that have a single true value*. If this procedure stops, then a choice for `p` has been found.

The following function carries out this procedure:


```python
def choice(p: Matrix) -> Vector:
    """Find a choice for a boolean matrix."""
    ch, a = np.empty((len(p), 2), dtype=int), 0
    indices = np.einsum("cab->abc", np.indices(p.shape))
    while len(p):
        unamb, amb = unambiguous(p)
        ch[a:(a := a + unamb.sum())] = indices[unamb]
        p, indices = p[amb], indices[amb]
    return ch[ch[:, 0].argsort(), 1]

def unambiguous(p: Matrix) -> Tuple[Matrix, Tuple]:
    """Unambiguous choices for a boolean matrix."""
    a = p.sum(axis=1) == 1
    b = p[a].sum(axis=0) == 1
    unamb = np.zeros(p.shape, dtype=bool)
    unamb[ab] = p[(ab := np.ix_(a, b))]
    amb = np.ix_(~a | ~p[:, b].any(axis=1), ~b)
    return unamb, amb  
```

<a id='day16'></a>
### [Day 16](https://adventofcode.com/2020/day/16): *Ticket Translation*

Let’s first shape the puzzle data into arrays.


```python
with open("data/16.txt", "r") as f:
    lines = [l.strip() for l in f.readlines()]
    ranges = re.findall(r"(\d+)-(\d+)", "".join(lines[:20]))
    lwr, upr = np.array(ranges, dtype=int).T.reshape(2, 20, 2)
    myticket = np.array(lines[22].split(","), dtype=int)
    tickets = np.array([t.split(",") for t in lines[25:]], dtype=int)
```

The rows of the interval-bound matrices `lwr`, `upr` are enumerated by the 20 ticket fields, while the rows of the ticket-number matrix `tickets` are enumerated by the tickets.


```python
print(f"{lwr.shape=}, {upr.shape=}, {tickets.shape=}")
```

    lwr.shape=(20, 2), upr.shape=(20, 2), tickets.shape=(237, 20)


For **Part 1**, we find that there are **`25788`** invalid ticket numbers, among all tickets.


```python
nums = tickets.reshape(tickets.size, 1, 1)
invalid = ((nums < lwr) | (upr < nums)).all(axis=(1, 2))

assert 25788 == np.sum(nums[invalid])
```

Part 2 amounts to finding the **order** in which the ticket fields correspond to the columns of `tickets`, once invalid tickets are discarded. But this order is precisely the sorting order of a **choice** for the relation between ticket numbers and ticket fields where a ticket number stands in relation to a ticket field whenever that ticket number, for all tickets, lies in the range of values for the ticket field.


```python
def maybefield(tickets: np.ndarray, lwr: np.ndarray, upr: np.ndarray) -> Matrix:
    """Incidence matrix for the aforementioned relation."""
    t = tickets.T[..., np.newaxis, np.newaxis]
    return ((lwr <= t) & (t <= upr)).any(axis=-1).all(axis=1)

valid = ~invalid.reshape(tickets.shape).any(axis=1)
enumfields = choice(maybefield(tickets[valid], lwr, upr))
```

For **Part 2**, we find that the product of the numbers in our ticket corresponding to the departure fields is **`3902565915559`**.


```python
departures = enumfields.argsort()[:6]  # first 6 field names starting with "departure"

assert 3902565915559 == np.prod(myticket[departures])
```

<a id='day21'></a>
### [Day 21](https://adventofcode.com/2020/day/21): *Allergen Assessment*

Let’s first collect the allergens and ingredients as lists of lists of strings, ordered by food.


```python
with open("data/21.txt", "r") as f:
    foods = re.findall(r"(.*) \(contains (.*)\)", f.read())
    foods = [(ing.split(), agn.split(", ")) for ing, agn in foods]
    ingredients, allergens = zip(*foods)
```

We can encode these lists of lists as incidence matrices with the use of the following function, `incidence()`, which returns an incidence matrix (`inc`) together with the corresponding array of sorted items (`xs`). These return values are characterized by the following property:

```python
inc, xs = incidence(xss)
assert [xs[i].tolist() for i in inc] == list(map(sorted, xss))
```

(In fact, we’ll need `inc` as an *integer* incidence matrix, which requires `xs[i.astype(bool)]` in place of `xs[i]`.)


```python
Itemlists = List[List[str]]

def incidence(xss: Itemlists) -> Tuple[Matrix, Vector]:
    """Integer incidence matrix and item array for lists of items."""
    xs = sorted(set(it.chain.from_iterable(xss)))
    enum = {x: i for i, x in enumerate(xs)}
    inc = np.zeros((len(xss), len(xs)), dtype=int)
    for i, xs_ in enumerate(xss):
        inc[i, [enum[x] for x in xs_]] = 1
    return inc, np.array(xs)
```

Unpacking `hasing, ing = incidence(ingredients)` assigns a (0,1)-matrix `hasing` (short for “has ingredient”) and an array `ing` of ingredient names in alphabetical order. Each row of `hasing` corresponds to a food, and tabulates the ingredients it contains. Ditto for `incidence(allergens)`.

By multiplying these incidence matrices, we can determine the incidence matrix for the **relation between allergens and ingredients** where an allergen is related to an ingredient if that ingredient is present whenever the allergen is. The resulting product tallies the number of foods in which an allergen and ingredient coincide, and the two are related if and only if that tally is maximal. Consequently, the following function computes the (boolean) incidence matrix between allergens and ingredients:


```python
def maybedangerous(hasagn: Matrix, hasing: Matrix) -> Matrix:
    """Incidence matrix for the aforementioned relation."""
    count = hasagn.T @ hasing
    return count == count.max(axis=1, keepdims=True)
```

(This implementation clarifies why `incidence()` returns an integer matrix: the matrix multiplication must be with integers, instead of booleans, in order for it to account for the *number*, and not simply the occurrence, of coincidences.)

Putting these functions together, we get a function `danger()` that computes a **choice** for the relation between allergens and ingredients (as well as the incidence matrix for ingredients and the ingredient names themselves). This choice is none other than a mapping from allergens to the ingredients containing them.


```python
def danger(allergens: Itemlists, ingredients: Itemlists):
    hasagn, _   = incidence(allergens)
    hasing, ing = incidence(ingredients)
    return choice(maybedangerous(hasagn, hasing)), hasing, ing
```

For **Part 1**, we find that **`1882`** ingredients appearing in the lists cannot possibly contain any of the allergens.


```python
dangerous, hasing, ing = danger(allergens, ingredients)

assert 1882 == hasing.sum() - hasing[:, dangerous].sum()
```

For **Part 2**, we find that the list of ingredients that containing allergens, sorted alphabetically by their allergen, is **`xgtj`**, **`ztdctgq`**, **`bdnrnx`**, **`cdvjp`**, **`jdggtft`**, **`mdbq`**, **`rmd`**, **`lgllb`**.


```python
canonicaldangerous = ",".join(ing[dangerous])

assert "xgtj,ztdctgq,bdnrnx,cdvjp,jdggtft,mdbq,rmd,lgllb" == canonicaldangerous
```

<a id='license'></a>
## License

Copyright 2021 Eugene Ha

The code is made available under the [MIT License](https://opensource.org/licenses/MIT); all other content is made available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
