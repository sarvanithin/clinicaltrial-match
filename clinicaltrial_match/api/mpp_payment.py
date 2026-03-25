"""
MPP (Machine Payments Protocol) setup for pay-per-query billing.

Charges $0.05 USDC (pathUSD on Tempo) per live-match query.
Enable by setting:
    CTM_MPP__ENABLED=true
    CTM_MPP__RECIPIENT_ADDRESS=0x<your-tempo-wallet>
    MPP_SECRET_KEY=<random-hex-32-bytes>   # generate: python3 -c "import secrets; print(secrets.token_hex(32))"

After deploying, run:
    npx -y @agentcash/discovery@latest discover <your-url>
and register on https://www.mppscan.com/register.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clinicaltrial_match.config import MPPConfig


def create_mpp(config: MPPConfig):  # type: ignore[return]
    """Build and return a configured Mpp instance, or None if MPP is disabled."""
    if not config.enabled or not config.recipient_address:
        return None

    secret_key = os.environ.get(config.secret_key_env, "")
    if not secret_key:
        raise ValueError(
            f"MPP is enabled but {config.secret_key_env} is not set. "
            'Generate one with: python3 -c "import secrets; print(secrets.token_hex(32))"'
        )

    from mpp.methods.tempo import ChargeIntent, tempo
    from mpp.server import Mpp

    return Mpp.create(
        method=tempo(
            currency=config.currency,
            recipient=config.recipient_address,
            intents={"charge": ChargeIntent()},
        ),
        secret_key=secret_key,
    )
