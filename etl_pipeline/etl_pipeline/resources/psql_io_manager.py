from contextlib import contextmanager
import pandas as pd
from dagster import IOManager, OutputContext, InputContext
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logs = logging.getLogger("psql_io_manager")


@contextmanager
def connect_psql(config, schema: str):
    conn_info = (
        f"postgresql+psycopg2://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )
    engine = create_engine(conn_info)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    yield engine


class PostgreSQLIOManager(IOManager):
    def __init__(self, config):
        self._config = config

    def load_input(self, context: InputContext) -> pd.DataFrame:
        raise NotImplementedError()

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        table = context.asset_key.path[-1]
        schema = context.asset_key.path[0]

        # lấy unique_key từ metadata của asset output
        unique_key = (context.output_metadata or {}).get("unique_key")

        # convert Dagster metadata value -> python object
        if unique_key is not None and hasattr(unique_key, "value"):
            unique_key = unique_key.value

        context.log.info(f"Writing to table {schema}.{table}")
        if unique_key:
            context.log.info(f"Dedup enabled with unique_key={unique_key}")

        with connect_psql(self._config, schema) as engine:
            with engine.begin() as conn:

                # 1) dedup trong dataframe
                if unique_key:
                    obj = obj.drop_duplicates(subset=unique_key, keep="last")

                if not unique_key:
                    # fallback: append bình thường
                    obj.to_sql(
                        table,
                        con=conn,
                        schema=schema,
                        if_exists="append",
                        index=False,
                    )
                    return

                # 2) tạo temp table
                temp_table = f"__tmp_{table}"
                conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{temp_table}"))

                obj.to_sql(
                    temp_table,
                    con=conn,
                    schema=schema,
                    if_exists="replace",
                    index=False,
                )

                # 3) delete record trùng key trong bảng thật
                join_cond = " AND ".join(
                    [f"t.{c} = s.{c}" for c in unique_key]
                )

                delete_sql = f"""
                    DELETE FROM {schema}.{table} t
                    USING {schema}.{temp_table} s
                    WHERE {join_cond}
                """
                conn.execute(text(delete_sql))

                # 4) insert dữ liệu mới
                insert_sql = f"""
                    INSERT INTO {schema}.{table}
                    SELECT * FROM {schema}.{temp_table}
                """
                conn.execute(text(insert_sql))

                # 5) drop temp
                conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{temp_table}"))

        context.log.info(f"Done write {schema}.{table}, rows={len(obj)}")