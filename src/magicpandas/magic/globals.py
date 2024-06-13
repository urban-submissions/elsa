
"""
The problem is that running in an interactive notebook the script
is more proactive in garbage collecting, so by the time you enter
raster.from_location(...).stitch(...), the root is already garbage collected.

from magicpandas.raster import Raster
stitched = (
    Raster
    .from_location(
        location='Harvard Yard, Cambridge, MA',
        source='Harvard Yard, Cambridge, MA',
        zoom=18,
    )
    .stitch(2)
)
"""
root = None
