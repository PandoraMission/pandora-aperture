# First-party/Local

# First-party/Local
import pandoraaperture as pa
from pandoraaperture import DOCSDIR


def test_prf():
    for cls in [pa.PRF, pa.SpatialPRF, pa.DispersedPRF]:
        prf = cls.from_reference()
        fig = prf.plot()
        fig.savefig(
            DOCSDIR + f"images/{cls.__name__}.png",
            dpi=150,
            bbox_inches="tight",
        )
        assert len(prf.evaluate()) == 3
