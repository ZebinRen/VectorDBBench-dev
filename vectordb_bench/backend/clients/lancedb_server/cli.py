from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import IndexType


class LanceDBServerTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="URI connection string", required=True),
    ]
    token: Annotated[
        str | None,
        click.option("--token", type=str, help="Authentication token", required=False),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBServerTypedDict)
def LanceDBServer(**parameters: Unpack[LanceDBServerTypedDict]):
    from .config import LanceDBServerConfig, _lancedb_server_case_config

    run(
        db=DB.LanceDBServer,
        db_config=LanceDBServerConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_server_case_config.get("NONE")(),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBServerTypedDict)
def LanceDBServerAutoIndex(**parameters: Unpack[LanceDBServerTypedDict]):
    from .config import LanceDBServerConfig, _lancedb_server_case_config

    run(
        db=DB.LanceDBServer,
        db_config=LanceDBServerConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=_lancedb_server_case_config.get(IndexType.AUTOINDEX)(),
        **parameters,
    )


class LanceDBServerIVFPQTypedDict(CommonTypedDict, LanceDBServerTypedDict):
    num_partitions: Annotated[int, click.option("--num-partitions", type=int, default=0, help="Number of partitions for IVFPQ index, unset = use LanceDB default")]
    num_sub_vectors: Annotated[int, click.option("--num-sub-vectors", type=int, default=0, help="Number of sub-vectors for IVFPQ index, unset = use LanceDB default")]
    nbits: Annotated[int, click.option("--nbits", type=int, default=8, help="Number of bits for IVFPQ index (must be 4 or 8), unset = use LanceDB default")]
    nprobes: Annotated[int, click.option("--nprobes", type=int, default=0, help="Number of probes for IVFPQ search, unset = use LanceDB default")]
    


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBServerIVFPQTypedDict)
def LanceDBServerIVFPQ(**parameters: Unpack[LanceDBServerIVFPQTypedDict]):
    from .config import LanceDBServerConfig, LanceDBServerIndexConfig

    run(
        db=DB.LanceDBServer,
        db_config=LanceDBServerConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        # db_case_config=_lancedb_server_case_config.get(IndexType.IVFPQ)(),
        db_case_config=LanceDBServerIndexConfig(
            index=IndexType.IVFPQ,
            num_partitions=parameters["num_partitions"],
            num_sub_vectors=parameters["num_sub_vectors"],
            nbits=parameters["nbits"],
            nprobes=parameters["nprobes"],
        ),  
        **parameters,
    )


class LanceDBServerHNSWTypedDict(CommonTypedDict, LanceDBServerTypedDict):
    m: Annotated[int, click.option("--m", type=int, default=0, help="HNSW parameter m")]
    ef_construction: Annotated[int, click.option("--ef-construction", type=int, default=0, help="HNSW parameter ef_construction")]
    ef: Annotated[int, click.option("--ef", type=int, default=0, help="HNSW search parameter ef")]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBServerHNSWTypedDict)
def LanceDBServerHNSW(**parameters: Unpack[LanceDBServerHNSWTypedDict]):
    from .config import LanceDBServerConfig, LanceDBServerHNSWIndexConfig

    run(
        db=DB.LanceDBServer,
        db_config=LanceDBServerConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        db_case_config=LanceDBServerHNSWIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef"],
        ),
        **parameters,
    )
