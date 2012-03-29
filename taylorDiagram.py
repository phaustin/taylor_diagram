#!/usr/bin/env python

__original_author__ = "Yannick Copin <yannick.copin@laposte.net>"

fontsize = 20.

import numpy as np
from tools.weighted_std import weighted_std
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import matplotlib.pyplot as plt

class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).

    # Initialize
    ------------
    # 1) create diagram (ref = observ)
    dia = TaylorDiagram(observ)
    # 2) create fig
    fignum = 1
    figsize = (15, 10)
    fig = plt.figure(num=fignum, figsize=figsize)
    plt.clf()
    # 3) create axes
    ax0 = dia.setup_axes(fig)
    # 4) create RMSD isolines
    ax0 = dia.add_rms_isolines()

    # Add new data (sample)
    ----------------------
    # 1) check size
    test = (np.size(observ) == np.size(sample))
    # 2) check formula 'RMS^2 - STD^2 - STD_ref^2 + 2*STD*STD_ref*COR'
    threshold = 1e-12
    test, value = dia.check_sample(sample, threshold)
    # 3) add on plot
    l = dia.plot_sample(key, sample, 'ro', markersize=5)
    
    # Display
    ---------
    ax0.legend(numpoints=1)
    plt.show()
    """

    def __init__(self, refsample, refweights=None, ref_label='Reference'):
        """refsample is the reference (data) sample to be compared to."""

        self.ref         = np.asarray(refsample.ravel())
        if refweights is None:
            self.weights = None
        else:
            self.weights = np.asarray(refweights.ravel())
        self.ref_mean    = np.ma.average(self.ref, weights=self.weights)
        self.ref_STD     = weighted_std(self.ref, weights=self.weights)
        self.ref_label   = ref_label
        
    def setup_axes(self, fig, rect=111, extra_graph=1.5, quadrant=1,
                   print_bool=False, *args, **kwargs):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes.

        Wouldn't the ideal be to define its own non-linear
        transformation, so that coordinates are directly r=stddev and
        theta=correlation? I guess it would allow 
        """

        self.extra_graph = extra_graph
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        if quadrant == 1:
            rlocs = np.concatenate((np.arange(0, 10, 1)/10.,
                                    [0.95, 0.99]))
            extremes = (0, np.pi/2, 0, extra_graph*self.ref_STD)
        elif quadrant == 2:
            rlocs = np.concatenate(([-0.99, -0.95],
                                    np.arange(-9, 10, 1)/10.,
                                    [0.95,0.99]))
            extremes = (0, np.pi, 0, extra_graph*self.ref_STD)
        else:
            raise ValueError("'quadrant' should be 1 or 2.")
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=extremes,
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].label.set_text("Standard deviation")
        ax.axis["right"].label._visible = True

        ax.axis["bottom"].set_visible(False)         # Useless
        
        # Grid
        ax.grid()

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        if print_bool:
            print "Reference std:", self.ref_STD
        self.ax.plot([0], self.ref_STD, 'bo',
                     label=self.ref_label,
                     markersize=2. * markersize)        
        self.ax.text(-0.045, self.ref_STD, "\\textbf{%s}" %self.ref_label,
                     color='b',
                     fontsize=fontsize,
                     horizontalalignment='left',
                     verticalalignment='top')
        if quadrant == 1:
            t = np.linspace(0., np.pi/2)
        elif quadrant == 2:
            t = np.linspace(0., np.pi)
        else:
            raise ValueError("'quadrant' should be 1 or 2.")
        r = np.zeros_like(t) + self.ref_STD
        self.ax.plot(t, r, 'k--')
        
        return ax

    def add_rms_isolines(self, num=8, *args, **kwargs):
        """Add RMS difference isolines to the graph, with values
        along curves. (Default = 8 isolines)"""
        
        t = np.linspace(0, np.pi/2., 1000)
        u = np.cos(t)
        rms_iso = self.ref_STD * np.arange(0., 1.701, 1.7/num)[1:]
        # 'line' on which will be added the RMS comments:
        v = self.extra_graph * self.ref_STD / (np.sin(t) +
                                               self.extra_graph*np.cos(t))
        a = 1
        b = -2. * self.ref_STD * u
        for rms in rms_iso:
            c = self.ref_STD**2 - rms**2
            delta = (b**2 - 4 * a * c)**0.5
            root1 = (-1. * b + delta) / (2. * a)
            root2 = (-1. * b - delta) / (2. * a)
            root1[root1 > self.extra_graph*self.ref_STD] = np.nan
            root2[root2 > self.extra_graph*self.ref_STD] = np.nan
            self.ax.plot(t, root1, 'g:')
            self.ax.plot(t, root2, 'g:')
            ind1 = np.nanargmin(abs(root1 - v))
            ind2 = np.nanargmin(abs(root2 - v))
            if ind1:
                ind = ind1
                root = root1
            elif ind2:
                ind = ind2
                root = root2
            else:
                raise ValueError("Problem in the RMSD legend process")
            self.ax.text(0.03 + t[ind], root[ind], "\\textbf{%s}" %("RMSD=%.2f" %rms),
                         color='g',
                         fontsize=0.80 * fontsize,
                         rotation=90 - (np.arctan(self.extra_graph/1.)
                                        *180./np.pi),
                         horizontalalignment='center',
                         verticalalignment='center')
        return self.ax

    def get_coords(self, name, sample, weights=None, print_bool=False):
        """Computes theta=arccos(correlation),rad=stddev of sample
        wrt. reference sample."""

        # flatten
        my_sample  = sample.ravel()
        if weights is None:
            my_weights = weights
        else:
            my_weights = weights.ravel()
        # (weighted) average and standard deviation of the sample
        ave = np.ma.average(my_sample, weights=my_weights)
        std = weighted_std(my_sample, weights=my_weights)
        # (weighted) correlation coefficient of the sample with the reference
        # after http://wapedia.mobi/en/Pearson_product-moment_correlation_coefficient?p=1
        # NO WEIGHTS: corr = np.corrcoef(self.ref, my_sample) # [[1,rho],[rho,1]]
        if my_weights is None:
            my_weights = np.ones_like(self.ref)
        cov_x_y = ( np.ma.sum( my_weights
                                * (my_sample - ave)
                                * (self.ref - self.ref_mean) )
                     / np.ma.sum( my_weights ) )
        cov_x_x = ( np.ma.sum( my_weights
                                * (my_sample - ave)**2)
                     / np.ma.sum( my_weights ) )
        cov_y_y = ( np.ma.sum( my_weights
                                * (self.ref - self.ref_mean)**2 )
                     / np.ma.sum( my_weights ) )
        corr    = 1. * cov_x_y / (cov_x_x * cov_y_y)**0.5
        theta = np.arccos(corr)
        ## # info to see how much does corr coeff change when use weighted corr vs non-weighted corr
        ## print '#' * 80
        ## mtxt = "corr WITHOUT weights = %.2f %%, corr WITH weights = %.2f %%, abs diff = %.2f %%, rel diff = %.2f %%, "
        ## no_weight_corr = np.corrcoef(self.ref, my_sample)[0,1]
        ## mtup = (100. * no_weight_corr,
        ##         100. * corr,
        ##         100. * abs(corr - no_weight_corr),
        ##         100. * 2 * abs(corr - no_weight_corr) / (corr + no_weight_corr)
        ##         )
        ## print mtxt %mtup
        ## print '#' * 80
        ## #
        if print_bool:
            print "std=%.2f and corr=%.2f"%(std, corr[0,1]), "for %s"%name
        return theta, std, corr

    def plot_sample(self, name, sample, weights=None, print_bool=False, *args, **kwargs):
        """Add sample to the Taylor diagram. args and kwargs are
        directly propagated to the plot command."""

        t, r, corr = self.get_coords(name, sample, weights, print_bool=print_bool)
        l, = self.ax.plot(t,r, *args, **kwargs) # (theta,radius)
        self.ax.text(-0.02 + t, 0.25 + r, "\\textbf{%s}" %name.replace('_', '\_'),
                     color='r',
                     fontsize=fontsize,
                     horizontalalignment='left',
                     verticalalignment='center')
        return l

    def check_sample(self, sample, weights=None, threshold=1e-12):
        """Check for the sample if the following relation holds:
        RMS^2 - STD^2 - STD_ref^2 + 2*STD*STD_ref*COR < threshold.
        """

        my_sample = sample.ravel()
        if weights is None:
            my_weights = weights
        else:
            my_weights = weights.ravel()
        means = np.ma.average(my_sample, weights=my_weights)
        STDs  = weighted_std(my_sample, weights=my_weights)
        terms = ((self.ref - self.ref_mean) - (my_sample - means))**2
        RMSs = (1. * np.sum(terms) / np.size(self.ref))**0.5
        # (weighted) correlation coefficient of the sample with the reference
        # after http://wapedia.mobi/en/Pearson_product-moment_correlation_coefficient?p=1
        # NO WEIGHTS: CORs = np.corrcoef(self.ref, my_sample)[0,1]  # [[1,rho],[rho,1]]
        if my_weights is None:
            my_weights = 1.
        cov_x_y = ( np.ma.sum( my_weights
                                * (my_sample - means)
                                * (self.ref - self.ref_mean) )
                     / np.ma.sum( my_weights ) )
        cov_x_x = ( np.ma.sum( my_weights
                                * (my_sample - means)**2)
                     / np.ma.sum( my_weights ) )
        cov_y_y = ( np.ma.sum( my_weights
                                * (self.ref - self.ref_mean)**2 )
                     / np.ma.sum( my_weights ) )
        CORs    = 1. * cov_x_y / (cov_x_x * cov_y_y)**0.5
        value = abs(RMSs - (STDs**2 + self.ref_STD**2 - 2*STDs*self.ref_STD*CORs)**0.5)
        test = value < threshold
        ## a = RMSs
        ## b = (STDs**2 + self.ref_STD**2 - 2*STDs*self.ref_STD*CORs)**0.5
        ## value1 = abs(2. * (a - b) / (a + b))
        ## value2 = abs(a - b)
        ## print "BITE, val = %.2e, val1 = %.2e, val2 = %.2e" %(value, value1, value2)
        return test, value

    def save_fig(self, fullName):
        """Save on disk the current Taylor diagram according to fullName."""

        self.ax.figure.savefig(fullName)
        #self.ax.figure.canvas.draw()
        #self.ax.figure.canvas.print_figure(fullName)

    def check_formula(self, dataset_dict, weights_dict=None, threshold=1e-12):
        """Check and print if the formula holds, i.e. if math OK."""
        
        print "Test : RMS^2 - STD^2 - STD_ref^2 + 2*STD*STD_ref*COR = 0"
        print "Threshold : %.2e" %(threshold)
        for src in dataset_dict:
            if weights_dict is not None:
                test, value = self.check_sample(dataset_dict[src], weights_dict[src], threshold)
            else:
                test, value = self.check_sample(dataset_dict[src], None, threshold)
            print test, '(' + ("%.2e" %value).rjust(9) + ") for %s" %src
        return None

    def add_datasets(self, dataset_dict, weights_dict=None, threshold=1e-12, fignum=1,
                     quadrant=0, print_bool=False, figsize=(14, 14), nb_rms_isolines=8):
        """Create figure, axes, RMSD isolines, and additional datasets."""
        
        # check datasets for quadrant
        neg = False
        for src in dataset_dict:
            if weights_dict is not None:
                test, value = self.check_sample(dataset_dict[src],
                                                weights_dict[src],
                                                threshold)
            else:
                test, value = self.check_sample(dataset_dict[src],
                                                None,
                                                threshold)
            if test:
                if weights_dict is None:
                    theta, std, corr = self.get_coords(src,
                                                       dataset_dict[src],
                                                       None,
                                                       print_bool=False)
                else:
                    theta, std, corr = self.get_coords(src,
                                                       dataset_dict[src],
                                                       weights_dict[src],
                                                       print_bool=False)
                if corr < 0:
                    neg = True
                    break                
        # create fig
        fig = plt.figure(num=fignum, figsize=figsize)
        plt.clf()
        # create axes
        if quadrant == 0 and neg:
            quadrant = 2
        elif quadrant == 0:
            quadrant = 1
        ax0 = self.setup_axes(fig, quadrant=quadrant, print_bool=False)
        # create RMSD isolines
        ax0 = self.add_rms_isolines(nb_rms_isolines)
        # add dataset_dict
        failed = []
        for src in dataset_dict:
            if weights_dict is not None:
                test, value = self.check_sample(dataset_dict[src],
                                                weights_dict[src],
                                                threshold)
            else:
                test, value = self.check_sample(dataset_dict[src],
                                                None,
                                                threshold)
            if test:
                if weights_dict is None:
                    l = self.plot_sample(src,
                                         dataset_dict[src],
                                         None,
                                         print_bool,
                                         'ro',
                                         markersize=1. * markersize,
                                         label=src.replace('_', '\_'))
                else:
                    l = self.plot_sample(src,
                                         dataset_dict[src],
                                         weights_dict[src],
                                         print_bool,
                                         'ro',
                                         markersize=1. * markersize,
                                         label=src.replace('_', '\_'))
            else:
                failed.append((src, value))
        return fig, ax0, failed


def local_plot(theLats, theLons, theData, plotopts, plot_type, title, cmap):
    # if lats decreasing, reverse it (and data too)
    if theLats[0,0] >= theLats[-1,-1]:
        lats = theLats[::-1]
        latRev = True
    else:
        lats = theLats[...]
        latRev = False
    # if lons decreasing, reverse it (and data too)
    if theLons[0,0] >= theLons[-1,-1]:
        lons = theLons[::-1]
        lonRev = True
    else:
        lons = theLons[...]
        lonRev = False
    # reverse data if needed
    if latRev and lonRev:
        data = theData[::-1,::-1]
    elif latRev:
        data = theData[::-1,...]
    elif lonRev:
        data = theData[...,::-1]
    else:
        data = theData[...]

    llcrnrlon = lons[ 0, 0] #ll corner at 0 degrees lon
    llcrnrlat = lats[ 0, 0] #ll corner at -90.0 lat
    urcrnrlon = lons[-1,-1] #ur corner at 357.5 lon
    urcrnrlat = lats[-1,-1] #ur corner at +90.0 lat
    # create the fig
    fignum=plotopts['fignum']
    theFig=plt.figure(fignum)
    theFig.clf()
    theAxis = theFig.add_axes([0.04,0.1,0.8,0.8],label='figure')
    m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
                projection='cyl', ax=theAxis, anchor='SW', resolution='c')
    width = plotopts['width']
    height=m.aspect*width
    theFig.set_size_inches(width,height*1.1)
    divider = make_axes_locatable(theAxis)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    m.drawcoastlines()
    x,y = m(lons,lats)
    #
    # norm=None means colormap will have min/max
    # limits of dataset
    #
    scale=plotopts['colorbar_scale']
    cextent=plotopts['colorbar_extend']
    if scale:
        norm = colors.normalize(scale[0], scale[1], clip=False)
    else:
        norm=None
    if cextent:
        extend=cextent
    else:
        extend='neither'
    if plot_type=='pcolormesh':
        im=m.pcolormesh(x,y,data,cmap=cmap,norm=norm)
    elif plot_type=='contourf':
        im=m.contourf(x,y,data,N,cmap=cmap,norm=norm)
    theFig.colorbar(im,format='%3g', cax=cax,extend=extend)
    color_axis = m.ax.figure.axes[1]
    cbar_label = color_axis.set_ylabel('degree Kelvin (K)')
    theFig.canvas.draw()
    m.ax.set_title(title)
    color_axis = m.ax.figure.axes[1]
    m.ax.figure.canvas.draw()
    m.ax.figure.canvas.print_figure(title.replace(' ', '_') + '.png', rasterized=True)
    m.ax.figure.canvas.print_figure(title.replace(' ', '_') + '.eps', rasterized=True)
    m.ax.figure.canvas.print_figure(title.replace(' ', '_') + '.pdf', rasterized=True)

    return m


if __name__=='__main__':
    """This main part illustrates how to use Taylor Diagram.

    See the class docstring for more details. Data and models are
    creating using create_data.py and create_models.py in order to
    illustrate at best the diagram.

    Other simple test:
    ------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from taylorDiagram import TaylorDiagram

    A = np.random.random((100, 100, 100))
    B = np.random.random((100, 100, 100))
    C = A + np.random.random((100, 100, 100))
    D = 0.5 * ( A + np.random.random((100, 100, 100)) )

    dia = TaylorDiagram(A)
    fignum = 1
    figsize = (15, 10)
    fig = plt.figure(num=fignum, figsize=figsize)
    plt.clf()
    ax0 = dia.setup_axes(fig)
    ax0 = dia.add_rms_isolines()

    threshold = 1e-12
    if (np.size(A) == np.size(B)) and \
       dia.check_sample(B, threshold)[0]:
        l = dia.plot_sample('B', B, 'ro', markersize=5)
    if (np.size(A) == np.size(C)) and \
       dia.check_sample(C, threshold)[0]:
        l = dia.plot_sample('C', C, 'ro', markersize=5)
    if (np.size(A) == np.size(D)) and \
       dia.check_sample(D, threshold)[0]:
        l = dia.plot_sample('D', D, 'ro', markersize=5)

    ax0.legend(numpoints=1)
    plt.show()
    """
    
    # <Imports>=
    # /home/ccorbel/repos/group/christophe_code/tools/create_data.py
    # /home/ccorbel/repos/group/christophe_code/tools/create_models.py
    from tools.create_grid import create_grid
    from tools.create_data import create_data
    from tools.create_models import create_models
    from tools.cmap_creator import cmap_creator
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=fontsize)
    rc('lines', linewidth=3)
       
    # <Values>=

    fontsize = 20.
    markersize = 15.
    fignum = 0
    # grid
    nlon = 144
    nlat = 72
    ntime = 36
    # model offsets
    off_lon = 18
    off_lat = 9
    off_time = 3
    # temperatures
    temp_min = -30 + 273.15
    temp_max =  30 + 273.15
    delta_time = 10
    # noise
    noise_ampli = 5.0
    noise_weak = 5.0
    noise_strong = 10.0

    # <Data and Models>=
    
    # grid
    lon, lat, theLons, theLats = create_grid(nlon, nlat, ntime,
                                             lat_start=80., lat_stop=-79.,
                                             lon_start=10., lon_stop=349.)
    # data
    observations, observ = create_data(nlon, nlat, ntime,
                                       off_lon, off_lat, off_time,
                                       temp_min, temp_max, delta_time,
                                       noise_ampli)
    # models
    mixing_unif = [0.91, 0.67, 0.33]
    models = create_models(nlon, nlat, ntime,
                           off_lon, off_lat, off_time,
                           mixing_unif, observations, observ,
                           noise_ampli, noise_weak, noise_strong)
    nmodel = len(models.keys())
    
    # check temperature field
    cmap = cmap_creator('GMT_polar')
    plot_type = 'pcolormesh'
    title     = 'Temperature field with latitude gradient'
    fignum   += 1
    plotopts  = {'fignum': fignum,
                 'width': 8.,
                 'colorbar_scale': None,
                 'colorbar_extend': None}
    m = local_plot(theLats, theLons, observ[0, ...], plotopts, plot_type, title, cmap)

    # check sizes
    test = True
    for key in models:
        test = test and np.size(observ) == np.size(models[key])
    if test:
        print "ok, data sets same size"
        print
    else:
        raise ValueError("NOT ok, data sets NOT same size")

    # <Taylor Diagram>=

    # create diagram
    weights = np.arange(1, np.product(np.shape(observ)) + 1).reshape(np.shape(observ))
    if hasattr(observ, 'mask'):
        weights = np.ma.array(data=weights,
                              mask=observ.mask,
                              dtype=weights.dtype)
    dia = TaylorDiagram(observ, ref_label='Reference')

    # Check formula
    threshold = 1e-1
    weights_dict = {}
    for key in models:
        weights_dict[key] = weights
    dia.check_formula(models, weights_dict, threshold=threshold)
    print

    # create fig
    fignum += 1
    figsize = 14.
    figsize = (figsize, figsize)
    nb_rms_isolines = 8.
    fig, ax0, failed = dia.add_datasets(models,
                                        weights_dict,
                                        threshold = threshold,
                                        fignum = fignum,
                                        quadrant = 1,
                                        print_bool=False,
                                        figsize = figsize,
                                        nb_rms_isolines = nb_rms_isolines)

    
    # edit
    blow = 0.07
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0 - 2. * blow * box.width,
                     box.y0 - 1. * blow * box.height,
                     box.width  * (1.00 + 2. * blow),
                     box.height * (1.00 + 2. * blow)])

    # plot
    #dia.ax.legend(numpoints=1, loc='best')
    plt.show()
    
    # save pdf
    figFolder = '' # 'figures/'
    figName   = 'taylor_diagram_example'
    figExt    = '.pdf'
    fullName  = figFolder + figName + figExt
    dia.save_fig(fullName)

    # save png
    figFolder = '' # 'figures/'
    figName   = 'taylor_diagram_example'
    figExt    = '.png'
    fullName  = figFolder + figName + figExt
    dia.save_fig(fullName)

    # save eps
    figFolder = '' # 'figures/'
    figName   = 'taylor_diagram_example'
    figExt    = '.eps'
    fullName  = figFolder + figName + figExt
    dia.save_fig(fullName)

