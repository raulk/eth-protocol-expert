"""Loader for ethereum/beacon-APIs specifications."""

from pathlib import Path

from .openapi_loader import OpenAPILoader


class BeaconAPIsLoader(OpenAPILoader):
    """Load beacon APIs from ethereum/beacon-APIs repository.

    The Beacon Node API enables users to query and participate in the Ethereum
    consensus layer (formerly Eth2). This loader parses the OpenAPI 3.1 spec
    and extracts endpoints for beacon chain state, validators, blocks, and more.

    The main spec file (beacon-node-oapi.yaml) references endpoints in:
    - apis/beacon/ - Beacon chain state and block endpoints
    - apis/config/ - Chain configuration endpoints
    - apis/debug/ - Debug endpoints (not for public exposure)
    - apis/eventstream/ - Server-sent events
    - apis/node/ - Node info endpoints
    - apis/validator/ - Validator client endpoints
    - types/ - Shared schema definitions across forks
    """

    REPO_URL = "https://github.com/ethereum/beacon-APIs.git"

    def __init__(self, repo_path: str | Path = "data/beacon-apis") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            source_name="beacon-apis",
            spec_file="beacon-node-oapi.yaml",
        )
