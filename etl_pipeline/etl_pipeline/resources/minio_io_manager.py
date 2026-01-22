from dagster import IOManager, OutputContext, InputContext
from minio import Minio
import pandas as pd
import os
import tempfile


class MinIOIOManager(IOManager):
    def __init__(self, config):
        self._config = config
        self.client = Minio(
            config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=config.get("secure", False),
            region="us-east-1",
        )
    
    def _resolve_object_path(self, asset_key, partition_key=None) -> str:
        """
        asset_key.path = [layer, schema?, table]
        """
        parts = asset_key.path

        layer = parts[0]
        table = parts[-1]
        prefix = f"{layer}_"
        if table.startswith(prefix):
            table = table[len(prefix):]

        schema = "/".join(parts[1:-1]) if len(parts) > 2 else ""

        base = f"{layer}/{schema}/{table}".replace("//", "/")

        if partition_key:
            return f"{base}/{partition_key}.parquet"

        return f"{base}.parquet"

    def _tmp_file(self, asset_key, partition_key=None):
        name = asset_key.path[-1]
        suffix = f"_{partition_key}" if partition_key else ""
        return os.path.join(
            tempfile.gettempdir(),
            f"{name}{suffix}.parquet",
        )

    # =====================================================
    # 🔹 OUTPUT
    # =====================================================
    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        if obj is None or obj.empty:
            context.log.info("No data to write")
            return

        partition_key = context.partition_key if context.has_partition_key else None

        object_name = self._resolve_object_path(
            context.asset_key, partition_key
        )
        tmp_path = self._tmp_file(context.asset_key, partition_key)

        obj.to_parquet(tmp_path, index=False)

        self.client.fput_object(
            bucket_name=self._config["bucket_name"],
            object_name=object_name,
            file_path=tmp_path,
            content_type="application/parquet",
        )

        context.log.info(f"Written to MinIO: {object_name}")
        os.remove(tmp_path)

    # =====================================================
    # 🔹 INPUT
    # =====================================================
    def load_input(self, context: InputContext) -> pd.DataFrame:
        metadata = context.metadata or {}
        load_all_partitions = metadata.get("load_all_partitions", False)

        # ==============================
        # CASE 3: partitioned → unpartitioned (merge)
        # PHẢI ƯU TIÊN
        # ==============================
        if load_all_partitions:
            partitions = metadata.get("partitions") or self.list_partitions(
                context.asset_key
            )

            dfs = []
            for pk in sorted(partitions):
                try:
                    df = self.load_partition(context.asset_key, pk)
                    df["_partition_key"] = pk
                    dfs.append(df)
                except FileNotFoundError:
                    continue

            if not dfs:
                raise FileNotFoundError(
                    f"No partitions found for asset {context.asset_key}"
                )

            return pd.concat(dfs, ignore_index=True)

        # ==============================
        # CASE 2: unpartitioned asset
        # ==============================
        if not context.has_partition_key:
            object_name = self._resolve_object_path(
                context.asset_key, partition_key=None
            )
            tmp_path = self._tmp_file(context.asset_key, None)

            try:
                self.client.fget_object(
                    bucket_name=self._config["bucket_name"],
                    object_name=object_name,
                    file_path=tmp_path,
                )
            except Exception:
                raise FileNotFoundError(object_name)

            df = pd.read_parquet(tmp_path)
            os.remove(tmp_path)
            return df

        # ==============================
        # CASE 1: partitioned → partitioned
        # + fallback to unpartitioned
        # ==============================
        try:
            return self.load_partition(
                context.asset_key, context.partition_key
            )
        except FileNotFoundError:
            # fallback: load unpartitioned asset
            object_name = self._resolve_object_path(
                context.asset_key, partition_key=None
            )
            tmp_path = self._tmp_file(context.asset_key, None)

            try:
                self.client.fget_object(
                    bucket_name=self._config["bucket_name"],
                    object_name=object_name,
                    file_path=tmp_path,
                )
            except Exception:
                raise FileNotFoundError(
                    f"No partition '{context.partition_key}' "
                    f"and no unpartitioned file for asset {context.asset_key}"
                )

            df = pd.read_parquet(tmp_path)
            os.remove(tmp_path)

            # optional: gắn partition_key hiện tại
            df["_partition_key"] = context.partition_key
            return df


    # =====================================================
    # 🔹 PARTITION HELPERS
    # =====================================================
    def list_partitions(self, asset_key):
        prefix = self._resolve_object_path(asset_key).replace(".parquet", "") + "/"

        objects = self.client.list_objects(
            self._config["bucket_name"],
            prefix=prefix,
            recursive=True,
        )

        partitions = []
        for obj in objects:
            if obj.object_name.endswith(".parquet"):
                partitions.append(
                    obj.object_name.split("/")[-1].replace(".parquet", "")
                )

        return sorted(partitions)

    def load_partition(self, asset_key, partition_key) -> pd.DataFrame:
        object_name = self._resolve_object_path(asset_key, partition_key)
        tmp_path = self._tmp_file(asset_key, partition_key)

        try:
            self.client.fget_object(
                self._config["bucket_name"],
                object_name,
                tmp_path,
            )
        except Exception:
            raise FileNotFoundError(object_name)

        df = pd.read_parquet(tmp_path)
        os.remove(tmp_path)
        return df
    
    def write_partition(self, asset_key, partition_key, df: pd.DataFrame):
        base_path = self._resolve_object_path(asset_key).replace(".parquet", "")
        object_name = f"{base_path}/{partition_key}.parquet"

        tmp_file = os.path.join(
            tempfile.gettempdir(),
            f"{asset_key.path[-1]}_{partition_key}.parquet"
        )

        df.to_parquet(tmp_file, index=False)

        self.client.fput_object(
            bucket_name=self._config["bucket_name"],
            object_name=object_name,
            file_path=tmp_file,
            content_type="application/parquet",
        )

        os.remove(tmp_file)
