from .utils import *
class ModelInit:
    __slots__ = ('_zone_name','_zone_width', '_zone_depth', '_zone_height',
                 '_U_factor', '_SHGC' ,'_wea_dir', '_working_dir', '_observers')

    def __init__(self, zone_name = None, zone_width = None, zone_depth = None,
                 zone_height = None,  U_factor = None, SHGC = None, wea_dir = None,
                 working_dir = None):
        self.zone_name = zone_name
        self.zone_width = zone_width
        self.zone_depth = zone_depth
        self.zone_height = zone_height
        self.U_factor = U_factor
        self.SHGC = SHGC
        self.wea_dir = wea_dir
        self.working_dir = working_dir
        self._observers = None

    @property
    def zone_name(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_name

    @zone_name.setter
    def zone_name(self, value):
        self._zone_name = value


    @property
    def zone_width(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_width

    @zone_width.setter
    def zone_width(self, value):
        self._zone_width = value

    @property
    def zone_depth(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_depth

    @zone_depth.setter
    def zone_depth(self, value):
        self._zone_depth = value

    @property
    def zone_height(self):
        """Get or set a boolean for whether the zone sizing calculation is run."""
        return self._zone_height

    @zone_height.setter
    def zone_height(self, value):
        self._zone_height = value

    @property
    def wea_dir(self):
        return self._wea_dir

    @wea_dir.setter
    def wea_dir(self,value):
        self._wea_dir = value

    @property
    def U_factor(self):
        return self._U_factor

    @U_factor.setter
    def U_factor(self, value):
        self._U_factor = value

    @property
    def SHGC(self):
        return self._SHGC

    @SHGC.setter
    def SHGC(self, value):
        self._SHGC = value



    @property
    def working_dir(self):
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    @property
    def viewfactor(self):
        assert self._observers is not None, 'No observer attached'
        checker = self._observers(self._zone_width, self._zone_depth, self._zone_height)
        if checker:
            print("rerunning vf cal")
        else:
            print("no need to rerun vf cal")


    def bind_to(self, callback):
        print('bound')
        self._observers = callback
    #         print(len(self._observers))
# TODO: NEED TO ADD IN OBSERVER

class observer(object):
    def __init__(self, data):
        self.width = None
        self.depth = None
        self.height = None
        self.data = data
        self.data.bind_to(self.check_update)

    def check_update(self, width, depth, height):  # If self.width and self.
        if (width != self.width) or (depth != self.depth) or (height != self.height):
            self.width = width
            self.depth = depth
            self.height = height
            return True
        else:
            return False

