import pandas as pd
import numpy as np
import pandas as pd
from fractions import Fraction
class Post:
    """
    This object is to keep all the essential data in one place.
    This object is isolated from other objects
    """
    def __init__(self, obj):
        self._obj = obj

    @property
    def seasonlabel(self):
        """
        Determine the season of the array of 8760 hours of the year
        :return:
        """
        allhours = pd.date_range(start='1900-01-01 00:00:00', end='1900-12-31 23:59:59', freq='H')
        allhours = (allhours.dayofyear - 1) * 24 + allhours.hour
        winter = pd.date_range(start='1900-12-21 00:00:00', end='1901-03-19 23:59:59', freq='H')
        winter = (winter.dayofyear - 1) * 24 + winter.hour
        winter = allhours.isin(winter)
        summer = pd.date_range(start='1900-06-21 00:00:00', end='1900-09-23 23:59:59', freq='H')
        summer = (summer.dayofyear - 1) * 24 + summer.hour
        summer = allhours.isin(summer)
        allother = ~(summer | winter)

        seasons = np.array([None] * 8760)
        seasons[winter] = 'Winter'
        seasons[summer] = 'Summer'
        seasons[allother] = 'Shoulder'
        return seasons

    @property
    def suns(self):
        suns = tuple(self._obj.initmodel.sunpath.calculate_sun_from_hoy(hoy) for hoy in range(8760))
        return suns


    @property
    def solarlabel(self):
        """
        Determine the solar condition of the 8760 hours in a year given the orientation of the zon
        :return:
        """
        # if direction == 'south':
        zenith = np.array([90 - s.altitude for s in self.suns])
        azimuth = np.array([s.azimuth for s in self.suns])
        azimuth = np.array(azimuth) - 180   # manually reverse the azimuth convention

        if self._obj.initmodel.orientation == 'south':
            cos_incidence = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth - 0))

        elif self._obj.initmodel.orientation == 'north':
            cos_incidence = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth - 180))

        elif self._obj.initmodel.orientation == 'east':
            cos_incidence = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth + 90))

        elif self._obj.initmodel.orientation == 'west':
            cos_incidence = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth - 90))
        else:
            cos_incidence = None

        incident = np.rad2deg(np.arccos(cos_incidence))

        boolarr = (incident < 90) & (zenith < 90)

        return_value =np.array([None] * 8760)
        return_value[boolarr], return_value[~boolarr] = 'Irr', 'Non-irr'
        return return_value


    @property
    def numwarmpts(self):
        """

        :return:
        """
        return (self._obj.draft_adjusted_PMV > 0.5).sum(axis=0)

    @property
    def numcoldpts(self):
        """

        :return:
        """
        return (self._obj.draft_adjusted_PMV < -0.5).sum(axis=0)

    @property
    def percentagewarm(self):
        """

        :return:
        """
        return self.numwarmpts / len(self._obj.initmodel.testPts2D)

    @property
    def percentagecold(self):
        """

        :return:
        """
        return self.numcoldpts / len(self._obj.initmodel.testPts2D)


    @property
    def avgwarmPPD(self):
        """

        :return:
        """
        return np.array([self._obj.draft_adjusted_PPD[:,hour_i][self._obj.draft_adjusted_PMV[:,hour_i] > 0.5].mean() for hour_i in range(8760)])


    @property
    def avgcoldPPD(self):
        """

        :return:
        """
        return np.array([self._obj.draft_adjusted_PPD[:,hour_i][self._obj.draft_adjusted_PMV[:,hour_i] < -0.5].mean() for hour_i in
                range(8760)])

    @property
    def labeleddf(self):
        """

        :return:
        """
        seasonlabel = self.seasonlabel
        solarlabel = self.solarlabel
        ratio = Fraction(self._obj.initmodel.zone_width / self._obj.initmodel.zone_depth).limit_denominator()
        df_dict ={
            'Season': seasonlabel,
            'Solar Condition': solarlabel,
            'Num Pts w/ PMV > 0.5': self.numwarmpts,
            'Num Pts w/ PMV < -0.5': self.numcoldpts,
            '$DP_{\mathrm{warm}}$': self.percentagewarm,
            '$DP_{\mathrm{cold}}$': self.percentagecold,
            '$TS_{\mathrm{warm}}$': self.avgwarmPPD, # Degree of discomfort
            '$TS_{\mathrm{cold}}$': self.avgcoldPPD, #
            'Orientation': self._obj.initmodel.orientation.capitalize(),
            'WWR': self._obj.initmodel.WWR,
            'U-factor': round(self._obj.initmodel.U_factor, 2),
            'SHGC': round(self._obj.initmodel.SHGC, 2),
            'AR': str(ratio.numerator) + ":" + str(ratio.denominator)
        }
        df = pd.DataFrame(df_dict)
        df['Season & solar'] = df['Season'] + ', ' + df['Solar Condition']

        ## Transform DP into 0-100

        df['$DP_{\mathrm{warm}}$'] = df['$DP_{\mathrm{warm}}$'] * 100
        df['$DP_{\mathrm{cold}}$'] = df['$DP_{\mathrm{cold}}$'] * 100

        return df


    @property
    def df_melt_TS(self):
        """

        :return:
        """
        return pd.melt(self.labeleddf, id_vars = ['Orientation',  'Season & solar', 'WWR', 'U-factor', 'SHGC', 'AR'], value_vars = ['$TS_{\mathrm{warm}}$', '$TS_{\mathrm{cold}}$'] , value_name  = 'Averaged $TS$')


    @property
    def df_melt_DP(self):
        """

        :return:
        """
        return pd.melt(self.labeleddf, id_vars=['Orientation', 'Season & solar', 'WWR', 'U-factor', 'SHGC', 'AR'],
                       value_vars=['$DP_{\mathrm{warm}}$', '$DP_{\mathrm{cold}}$'], value_name='Averaged $DP$')

    @property
    def summary(self):
        pass


# Below are for showing intermediate results.... They can be used for debuging purpose
    @property
    def zerothDP(self):  # This is the DP with nothing considered

        return ((self._obj.zerothPMV > 0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)).mean() * 100, ((self._obj.zerothPMV < -0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)).mean() * 100


    @property
    def firstDP(self):   # This is the DP with solar radiation incorporated

        return ((self._obj.unadjustedPMV > 0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)).mean() * 100, ((self._obj.unadjustedPMV < -0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)).mean() * 100

    @property
    def zerothDP8760(self):
        return ((self._obj.zerothPMV > 0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)) * 100, ((self._obj.zerothPMV < -0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)) * 100


    @property
    def firstDP8760(self):
        return ((self._obj.unadjustedPMV > 0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)) * 100, ((self._obj.unadjustedPMV < -0.5).sum(axis=0) / len(self._obj.initmodel.testPts2D)) * 100


    @property
    def finalDP(self):

        return self.percentagewarm.mean() * 100, self.percentagecold.mean() * 100

    @property
    def zerothTS(self):
        warm_8760  = np.array([self._obj.zerothPPD[:,hour_i][self._obj.zerothPMV[:,hour_i] > 0.5].mean() for hour_i in range(8760)])
        cold_8760  = np.array([self._obj.zerothPPD[:,hour_i][self._obj.zerothPMV[:,hour_i] < -0.5].mean() for hour_i in range(8760)])
        return warm_8760, cold_8760

    @property
    def firstTS(self):
        warm_8760  = np.array([self._obj.firstPPD[:,hour_i][self._obj.unadjustedPMV[:,hour_i] > 0.5].mean() for hour_i in range(8760)])
        cold_8760  = np.array([self._obj.firstPPD[:,hour_i][self._obj.unadjustedPMV[:,hour_i] < -0.5].mean() for hour_i in range(8760)])
        return warm_8760, cold_8760


    @property
    def finalTS(self):
        warm_8760 = np.array([self._obj.draft_adjusted_PPD[:,hour_i][self._obj.draft_adjusted_PMV[:,hour_i] > 0.5].mean() for hour_i in range(8760)])
        cold_8760 = np.array([self._obj.draft_adjusted_PPD[:,hour_i][self._obj.draft_adjusted_PMV[:,hour_i] < -0.5].mean() for hour_i in range(8760)])
        return warm_8760, cold_8760
