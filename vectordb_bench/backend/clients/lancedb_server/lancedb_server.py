import logging
from contextlib import contextmanager
import requests, json, numpy as np

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel

from ..api import IndexType, VectorDB
from .config import LanceDBServerConfig, LanceDBServerIndexConfig

log = logging.getLogger(__name__)


class VectorModel(LanceModel):
    id: int
    vector: list[float]


class LanceDBServer(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: LanceDBServerConfig,
        db_case_config: LanceDBServerIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "LanceDBServer"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.uri = db_config["uri"]
        # avoid the search_param being called every time during the search process
        self.search_config = db_case_config.search_param()
        
        log.info(f"Search config: {self.search_config}")

        if drop_old:
            raise NotImplementedError("Drop old table is not implemented for LanceDBServer")

    @contextmanager
    def init(self):
        self.db = 123 #lancedb.connect(self.uri)
        self.table = 456 #self.db.open_table(self.table_name)
        yield
        self.db = None
        self.table = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
    ) -> tuple[int, Exception | None]:
        
        raise NotImplementedError("Insert method is not implemented for LanceDBServer")
        # try:
        #     data = [{"id": meta, "vector": emb} for meta, emb in zip(metadata, embeddings, strict=False)]
        #     self.table.add(data)
        #     return len(metadata), None
        # except Exception as e:
        #     log.warning(f"Failed to insert data into LanceDB table ({self.table_name}), error: {e}")
        #     return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        if filters is not None:
            raise NotImplementedError("Filters are not implemented for LanceDBServer")
        
        payload = {
            "query": query,
            "k": k,
        }
        if self.case_config.index == IndexType.IVFPQ and 'nprobes' in self.search_config.keys():
            payload["nprobes"] = self.search_config["nprobes"]
        elif self.case_config.index == IndexType.HNSW and 'ef' in self.search_config.keys():
            payload["ef"] = self.search_config['ef']
        else:
            raise NotImplementedError('search parameters must be provided')
        
        result = requests.post(self.uri+'/search', json=payload, timeout=30)
        result.raise_for_status()
        result = result.json()
        res = [r["id"] for r in result]
    
        return res
        
        if filters:
            results = (
                self.table.search(query)
                .select(["id"])
                .where(f"id >= {filters['id']}", prefilter=True)
                .limit(k)
            )
            if self.case_config.index == IndexType.IVFPQ and 'nprobes' in self.search_config.keys():
                results = results.nprobes(self.search_config["nprobes"]).to_list()
            elif self.case_config.index == IndexType.HNSW and 'ef' in self.search_config.keys():
                results = results.ef(self.search_config['ef']).to_list()
            else:
                results = results.to_list()
        else:
            results = self.table.search(query).select(["id"]).limit(k)
            if self.case_config.index == IndexType.IVFPQ and 'nprobes' in self.search_config.keys():
                results = results.nprobes(self.search_config["nprobes"]).to_list()
            elif self.case_config.index == IndexType.HNSW and 'ef' in self.search_config.keys():
                results = results.ef(self.search_config['ef']).to_list()
            else:
                results = results.to_list()
    
        return [int(result["id"]) for result in results]

    def optimize(self, data_size: int | None = None):
        raise NotImplementedError("Optimize method is not implemented for LanceDBServer")
        if self.table and hasattr(self, "case_config") and self.case_config.index != IndexType.NONE:
            log.info(f"Creating index for LanceDB table ({self.table_name})")
            log.info(f"Index parameters: {self.case_config.index_param()}")
            self.table.create_index(**self.case_config.index_param())
            # Better recall with IVF_PQ (though still bad) but breaks HNSW: https://github.com/lancedb/lancedb/issues/2369
            if self.case_config.index in (IndexType.IVFPQ, IndexType.AUTOINDEX):
                self.table.optimize()
