# polarq

**q/kdb+ semantics on Polars + Arrow Flight — runs natively in CPython.**

polarq is a Python library that brings the expressive power of q/kdb+ to the Python ecosystem. It works in two complementary ways: as a **transpiler** that compiles `.q` source files into native Python bytecode, and as a **runtime library** that backs that generated code with Polars-accelerated implementations of q's type system, verbs, adverbs, and table operations.

The result is q code that runs inside a standard Python process, debuggable with standard Python tools, interoperable with any Python library, and deployable anywhere Python runs.

---

## Design goals

**1. q semantics, Python runtime.**
polarq aims to be a faithful implementation of q's evaluation model — rank-polymorphic verbs, right-to-left expression evaluation, adverbs as higher-order functions, qSQL as a first-class query language — compiled down to CPython bytecode. The goal is not a toy subset; it is a platform capable of running real q scripts written for production kdb+ systems.

**2. Polars as the execution engine.**
Every operation that touches vectors or tables is routed through Polars. Polars `Series` backs `QVector`, `LazyFrame` backs `QTable`, and qSQL compiles directly to Polars method chains rather than being interpreted row by row. The Python escape hatch (`map_elements`, pure Python loops) exists for genuinely unrepresentable operations, but is never the first choice.

**3. Zero-cost q/Python interoperability.**
Because transpiled q code is just Python, a q function and a Python function can call each other without any FFI boundary, data copying, or format conversion. Python developers can contribute to q codebases. q logic can be incrementally replaced with idiomatic Python. The two languages share a single namespace and a single process.

**4. Arrow Flight as the IPC layer.**
Rather than implementing kdb+'s proprietary wire format, polarq builds its inter-process communication on Apache Arrow Flight. Arrow's columnar format is a zero-copy match for Polars internals, and Flight's gRPC-based protocol provides schema negotiation, backpressure, and cross-language reach that the kdb+ wire format lacks.

**5. Legacy q code as readable Python.**
The transpiler can emit `.py` source via `ast.unparse()` in addition to bytecode. This means a `.q` file can be converted to a human-readable Python file as a migration artefact, making legacy kdb+ systems accessible to Python developers without requiring them to learn q.

---

## Architecture

```
q source (.q / .k)
        │
        ▼
polarq.parser          Lark tokeniser + hand-written Pratt parser for expressions
        │              Produces a q-specific AST
        ▼
polarq.transpiler      Walks q AST, emits Python `ast` nodes
        │              Hoists multi-statement lambdas to `def` blocks
        │              Compiles qSQL to Polars LazyFrame method chains
        │              Produces ast.Module → compile() → CPython bytecode
        │              Optionally emits .py source via ast.unparse()
        ▼
CPython bytecode       Runs natively — standard debugger, profiler, import system
        │
        ▼
polarq runtime         All generated code imports from here:
  ├── types            QValue type tower (QAtom, QVector, QTable, ...)
  ├── env              Namespace chain (QEnv), dotted namespace support
  ├── coerce           Rank-polymorphism, type promotion lattice
  ├── verbs            Primitive verb dispatch (+  -  *  %  &  |  ~  # ...)
  ├── adverbs          Higher-order primitives (over, scan, each, each_left, each_right)
  ├── tables           qSQL → Polars LazyFrame compiler (select, update, exec, delete)
  ├── temporal         q epoch timestamps, date/time arithmetic, nanosecond precision
  ├── ipc              Arrow Flight server and client, pub/sub fanout
  ├── system           .z namespace stubs, \t timing, \l load, \d directory
  ├── stdlib           Named functions: aj, wj, lj, ij, string, parse, ...
  └── errors           QTypeError, QLengthError, QRankError, QDomainError
        │
        ▼
Polars + PyArrow        LazyFrame, Series, join_asof, Arrow IPC, Flight RPC
```

### Execution routing

Every operation follows a strict priority order designed to keep as much work as possible in Polars:

```
Incoming operation
        │
        ├─ qSQL statement?
        │    └─ compile to LazyFrame chain (zero Python overhead)
        │
        ├─ Operating on QTable columns?
        │    └─ try _ast_to_polars_expr()
        │         └─ fallback: map_elements (row-level Python)
        │
        ├─ Operating on QVector?
        │    └─ try Series vectorised op
        │         └─ fallback: Python-level map
        │
        └─ Scalar / atom
             └─ pure Python
```

The fallback paths exist for correctness; the Polars paths are the performance contract.

---

## Library layout

```
polarq/
├── __init__.py          Public API surface: Q session class, qvec(), re-exports
│
├── parser/
│   ├── lexer.py         Lark grammar + token definitions
│   ├── pratt.py         Hand-written Pratt parser for q expressions
│   │                    Handles rank-ambiguous verbs, adverb postfix, juxtaposition
│   ├── ast_nodes.py     q AST dataclasses (IntAtom, QSelect, Lambda, Adverb, ...)
│   └── qsql.py          Sub-grammar for select / update / exec / delete
│
├── transpiler/
│   ├── transpiler.py    Main QToPythonTranspiler — walks q AST, emits Python ast nodes
│   ├── expr.py          q expression → pl.Expr compiler (used inside qSQL)
│   ├── builtins.py      Maps q verb tokens to polarq.verbs.VERB_TABLE references
│   └── loader.py        load_q(path, env) — parse + transpile + exec a .q file
│
├── types.py             QValue type tower
│                          QNull, QAtom, QVector, QList
│                          QDict, QTable, QKeyedTable
│                          QLambda, QPartial, QBuiltin, QAdverb
│
├── env.py               QEnv — linked scope chain, dotted namespace (.myns.foo),
│                          global vs local assignment, child scope creation
│
├── coerce.py            Rank-polymorphism and type promotion
│                          promote(x, y) — atom↔vector broadcast
│                          unify_kind()  — q type lattice (b→i→j→f→...)
│                          to_polars_expr() — QValue → pl.Expr for column context
│
├── verbs.py             All q primitive verbs as QBuiltin instances
│                          Arithmetic:  q_add  q_sub  q_mul  q_div
│                          Comparison:  q_lt  q_gt  q_eq  q_not
│                          List:        q_count  q_first  q_last  q_reverse
│                                       q_where  q_distinct  q_group
│                          Aggregation: q_sum  q_min  q_max  q_avg  q_dev  q_med
│                          VERB_TABLE:  dict mapping q token strings → QBuiltin
│
├── adverbs.py           Higher-order verb combinators
│                          over(verb, x)           — fold (+/ → pl.sum() etc.)
│                          scan(verb, x)           — running fold (+\ → pl.cum_sum())
│                          each(verb, x)           — map (f' → map_elements)
│                          each_left(verb, x, y)   — x f\: y
│                          each_right(verb, x, y)  — x f/: y
│                          each_both(verb, x, y)   — x f': y (paired)
│
├── tables.py            qSQL → Polars LazyFrame compiler
│                          compile_select(table, cols, where, by)
│                          compile_update(table, cols, where)
│                          compile_aj(by_cols, t1, t2)  → join_asof
│                          compile_wj(windows, by_cols, t1, t2, aggs)
│                          _to_expr(node) → pl.Expr  (recursive expression compiler)
│
├── temporal.py          q epoch handling and temporal arithmetic
│                          Q epoch: 2000-01-01 (vs Unix 1970-01-01)
│                          from_q_timestamp(ns) / to_q_timestamp(dt)
│                          timestamp_series(s)  — wraps pl.Datetime("ns") Series
│                          extract(attr, vec)   — year/month/dd/hh/mm/ss/ns
│
├── ipc.py               Arrow Flight IPC layer
│                          PolarQServer  — Flight server over QEnv
│                            do_get()    — eval q expr, stream result as record batches
│                            do_put()    — receive batches, update named table
│                            subscribe() — register callback for table updates (fanout)
│                          PolarQClient
│                            get(query)  → QTable   (zero-copy via Arrow)
│                            put(name, qtable)
│
├── system.py            q system namespace and metacommands
│                          .z.p  — current timestamp
│                          .z.h  — hostname
│                          .z.i  — process id
│                          \t expr   — time expression (μs)
│                          \l path   — load script
│                          \d ns     — set default namespace
│
├── stdlib.py            Named library functions beyond the primitive verbs
│                          Joins:     lj  ij  uj  aj  wj
│                          String:    string  parse  upper  lower  trim  like
│                          Math:      prd  msum  mcount  mavg  mdev  ratios  deltas
│                          Utility:   flip  raze  enlist  cross  rotate  fills
│
├── errors.py            Exception hierarchy
│                          QError (base)
│                          ├── QTypeError    — type mismatch
│                          ├── QLengthError  — vector length mismatch
│                          ├── QRankError    — wrong number of arguments
│                          └── QDomainError  — value out of domain
│
└── repl.py              Interactive shell
                           Reads q expressions line by line
                           \\ to drop into Python REPL
                           \t, \l, \d metacommands
                           Error messages with q-style context hints
```

---

## Quick start

```python
from polarq import Q, qvec
import polars as pl

q = Q()

# Vectors and arithmetic
prices = qvec(150.0, 280.0, 320.0, 95.0)
qtys   = qvec(100, 200, 150, 400)

# Adverbs
total_value = q.over(q.verbs["*"], prices * qtys)   # +/ of (price * qty)

# Tables
trade = q.table(pl.DataFrame({
    "sym":   ["AAPL", "GOOG", "AAPL", "MSFT"],
    "price": [150.0,  280.0,  155.0,  320.0],
    "qty":   [100,    200,    150,    400],
}))

# qSQL
result = q.select(trade,
    cols=[("sum_qty", q.sum("qty")), ("avg_px", q.avg("price"))],
    by=["sym"]
)
print(result.frame.collect())
```

```python
# Transpile a .q file directly
from polarq.transpiler import load_q

q = Q()
load_q("strategy.q", q.env)   # all definitions land in q.env

# Call a q function from Python
vwap_fn = q.env.get("vwap")
result  = vwap_fn(prices, qtys)
```

```python
# Arrow Flight server — expose tables over the network
from polarq.ipc import PolarQServer

server = PolarQServer(q.env, port=5010)
server.serve()   # blocks; tables in q.env are queryable by any Flight client
```

```python
# Arrow Flight client — query from another process or language
from polarq.ipc import PolarQClient

client = PolarQClient(port=5010)
trades = client.get("select from trade where sym=`AAPL")
```

---

## Transpiler output example

Given `strategy.q`:

```q
trade:([]sym:`AAPL`GOOG`MSFT;price:150.0 280.0 320.0;qty:100 200 150)
vwap:{sum[x*y]%sum y}
result:select sum qty,avg price by sym from trade where price>200.0
```

Running `polarq.transpiler.to_source("strategy.q")` emits:

```python
import polars as pl
from polarq import *

trade = QTable.from_dict({
    "sym":   ["AAPL", "GOOG", "MSFT"],
    "price": [150.0, 280.0, 320.0],
    "qty":   [100, 200, 150],
})

def _qfn_1(x, y):
    return over(q_add, x * y) / over(q_add, y)
vwap = _qfn_1

result = (trade
    .frame
    .filter(pl.col("price") > 200.0)
    .group_by("sym")
    .agg([pl.col("qty").sum(), pl.col("price").mean()])
)
result = QTable(result)
```

This is valid, lint-clean Python that any Python developer can read, modify, and extend.

---

## IPC architecture

polarq replaces kdb+'s proprietary wire format with Apache Arrow Flight. The mapping onto q's process model is direct:

```
kdb+ concept                polarq equivalent
─────────────────────────────────────────────────────────────────
Tickerplant (TP)            PolarQServer receiving do_put() streams
RDB (in-memory subscriber)  PolarQServer with subscribe() callback
HDB (on-disk history)       Parquet files read via pl.scan_parquet()
.z.pg (sync query handler)  FlightServer.do_get()
.z.ps (async publisher)     FlightServer.do_put()
h: hopen `::5010            PolarQClient("localhost", 5010)
h "select from trade"       client.get("select from trade")
neg[h] (`upd; data)         client.put("trade", qtable)
```

Arrow Flight carries Arrow record batches natively. Because `QTable.frame` is a Polars `LazyFrame` and Polars speaks Arrow natively, the data path from query result to wire is zero-copy in both directions.

---

## Development roadmap

| Phase | Scope |
|---|---|
| 1 | `types` + `errors` + `coerce` — type tower |
| 2 | `verbs` — arithmetic, comparison, list primitives |
| 3 | `adverbs` — over, scan, each, each_left, each_right |
| 4 | `tables` — qSQL select / update / where / by |
| 5 | `parser` + `transpiler` — .q files execute natively |
| 6 | `temporal` — timestamp arithmetic, nanosecond precision, aj/wj |
| 7 | `ipc` — Arrow Flight server and client, pub/sub fanout |
| 8 | `system` + `repl` — .z namespace stubs, interactive shell |

---

## Dependencies

| Package | Role |
|---|---|
| `polars` | Vector and table execution engine |
| `pyarrow` | Arrow IPC format, Flight RPC protocol |
| `lark` | Tokenisation and outer grammar for the q parser |
| `numpy` | Fallback for atom-level numeric operations |

Python 3.11 or later is required (uses `match` statements throughout, `slots=True` dataclasses, and `ast.unparse()`).

---

## Relationship to existing q/Python bridges

| Project | Approach | polarq difference |
|---|---|---|
| `pykx` | Embeds a real kdb+ process | No kdb+ licence required; pure Python |
| `qpython` | Serialises Python ↔ q wire format | No kdb+ process at all |
| `pyq` | C extension embedding q in Python | Runs without any q installation |

polarq is the only approach that treats q as a **source language to compile**, rather than a runtime to embed or a wire format to speak. The output runs anywhere Python runs, with no kdb+ dependency of any kind.

---

## License

MIT
