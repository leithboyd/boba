"""Cross-feature contract tests — invariants EVERY registered feature must satisfy (a declared mirror,
a set param_kind). Per-feature behaviour (parity / independent oracle / mirror commutation / span=1)
lives in the sibling `test_<feature>.py` files; the engine tests live in tests/research/test_selection.py and
tests/research/test_screening.py."""
import boba.features.flow_persistence            # noqa: F401  (registers)
import boba.features.ofi_ema                     # noqa: F401  (registers)
import boba.features.ofi_fast_slow               # noqa: F401  (registers)
import boba.features.price_dislocation           # noqa: F401  (registers)
import boba.features.price_momentum              # noqa: F401  (registers)
import boba.features.stoikov_premium_fast_slow   # noqa: F401  (registers)
import boba.features.trade_flow_imbalance        # noqa: F401  (registers)
from boba.features import base
from boba.features.base import ParamKind


def test_every_registered_feature_declares_mirror():
    # The mirror augmentation is a required invariant: every feature must DEFINE its reflection.
    for spec in base.all_specs():
        assert spec.mirror is not None, f"feature {spec.name!r} has no mirror augmentation defined"


def test_every_registered_feature_declares_param_kind():
    # param_kind drives the (1-D vs 2-D) span sweep in the shared screening/finalize notebooks.
    expected = {
        "price_dislocation": ParamKind.FAST_SLOW,
        "ofi_fast_slow": ParamKind.FAST_SLOW,
        "ofi_ema": ParamKind.SINGLE,
        "stoikov_premium_fast_slow": ParamKind.FAST_SLOW,
        "trade_flow_imbalance": ParamKind.SINGLE,
        "price_momentum": ParamKind.SINGLE,
        "flow_persistence": ParamKind.SINGLE,
    }
    for name, kind in expected.items():
        assert base.get(name).param_kind is kind, f"{name} param_kind mismatch"
    # and every registered feature has *some* param_kind set
    for spec in base.all_specs():
        assert isinstance(spec.param_kind, ParamKind)
