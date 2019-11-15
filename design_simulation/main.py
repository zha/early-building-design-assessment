class Model(object):
    __slots__ = ()
    def __init__(self, zone_name, zone_width, zone_depth, zone_height, U_factor, SHGC, working_dir):

        self.zone_name = zone_name
        self.zone_width = zone_width
        self.zone_depth = zone_depth
        self.zone_height = zone_height
        self.U_factor = U_factor
        self.SHGC = SHGC,
        self.working_dir = working_dir
    @property
    def zone_width(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_width
    @zone_width.setter
    def zone_width(self, value):
        self._zone_width = value

