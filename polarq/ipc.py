import pyarrow as pa
import pyarrow.flight as flight
from polarq.types import QTable, QVector, QAtom
from polarq.env import QEnv
import polars as pl
import threading

class PolarQServer(flight.FlightServerBase):
    """
    Arrow Flight server exposing QTables as named endpoints.
    Subscriptions handled via threading.Event + registered callbacks.
    """

    def __init__(self, env: QEnv, host="localhost", port=5010):
        super().__init__(f"grpc://{host}:{port}")
        self.env  = env
        self._subs: dict[str, list] = {}   # table_name → [callback, ...]
        self._lock = threading.Lock()

    # ── Query (sync get) ───────────────────────────────────────────────────────

    def get_flight_info(self, ctx, descriptor):
        table_name = descriptor.command.decode()
        qtable = self.env.get(table_name)
        if not isinstance(qtable, QTable):
            raise flight.FlightServerError(f"{table_name} is not a table")
        schema = qtable.frame.schema.to_arrow()
        endpoint = flight.FlightEndpoint(descriptor.command, [])
        return flight.FlightInfo(schema, descriptor, [endpoint], -1, -1)

    def do_get(self, ctx, ticket):
        """Execute a q expression and stream result as Arrow record batches."""
        query = ticket.ticket.decode()
        # If it's a table name, return it directly
        # If it's a q expression, eval it first
        try:
            result = self.env.get(query)
        except KeyError:
            from polarq.transpiler import quick_eval
            result = quick_eval(query, self.env)

        if not isinstance(result, QTable):
            raise flight.FlightServerError("result is not a table")

        df     = result.frame.collect()
        table  = df.to_arrow()
        return flight.RecordBatchStream(table)

    # ── Publish (async put) ───────────────────────────────────────────────────

    def do_put(self, ctx, descriptor, reader, writer):
        """Receive a stream of record batches and update a named table."""
        table_name = descriptor.command.decode()
        batches = []
        for chunk in reader:
            batches.append(chunk.data)
        arrow_table = pa.Table.from_batches(batches)
        df = pl.from_arrow(arrow_table)
        qtable = QTable(df.lazy())
        self.env.set_global(table_name, qtable)
        self._fanout(table_name, qtable)

    # ── Subscription / fanout (tickerplant pattern) ───────────────────────────

    def subscribe(self, table_name: str, callback):
        with self._lock:
            self._subs.setdefault(table_name, []).append(callback)

    def _fanout(self, table_name: str, qtable: QTable):
        with self._lock:
            callbacks = list(self._subs.get(table_name, []))
        for cb in callbacks:
            try: cb(table_name, qtable)
            except Exception as e:
                print(f"[polarq] subscriber error: {e}")


class PolarQClient:
    """Thin Arrow Flight client. Zero-copy: results come back as QTable."""

    def __init__(self, host="localhost", port=5010):
        self._client = flight.connect(f"grpc://{host}:{port}")

    def get(self, query: str) -> QTable:
        ticket  = flight.Ticket(query.encode())
        reader  = self._client.do_get(ticket)
        arrow   = reader.read_all()
        df      = pl.from_arrow(arrow)
        return QTable(df.lazy())

    def put(self, table_name: str, qtable: QTable):
        descriptor = flight.FlightDescriptor.for_command(table_name.encode())
        df    = qtable.frame.collect()
        table = df.to_arrow()
        writer, _ = self._client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()

