import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection  
from matplotlib.collections import LineCollection  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize 
import numpy as np
import os


def mean_normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data


def plot_morph_waveform(cell, electrode, lfp=None, t=None, x_zoom=1, z_zoom=1):    
    if lfp is None:
        #normalize to min peak    
        LFPmin = electrode.data.min(axis=1)    
        LFPnorm = -(electrode.data.T / LFPmin).T    
        ind = np.where(electrode.data == electrode.data.min())[0][0]    
        LFPtrace = electrode.data[ind, ]    
        t = cell.tvec    
    else:    
        LFPnorm  = mean_normalize(lfp)
        LFPmin = lfp.min(axis=1)    
        # LFPnorm = -(lfp.T / LFPmin).T    
        ind = np.where(lfp == lfp.min())[0][0]    
        LFPtrace = lfp[ind, ]    
        t = t 
    
    LFPmin = np.where(LFPmin < 0, LFPmin, np.nan)  
    log_LFPmin = np.log10(-LFPmin) 

    log_LFPmin[np.isnan(log_LFPmin)] = np.nanmin(log_LFPmin)  
    norm = Normalize(vmin=np.nanmin(log_LFPmin), vmax=np.nanmax(log_LFPmin))

    fig = plt.figure(dpi=160)    
    ax1 = fig.add_axes([0, 0, 1, 1], frameon=False)
        
    cax = fig.add_axes([0.6, 0.01, 0.4, 0.015])  

    ax1.plot(electrode.x, electrode.z, 'o', markersize=1, color='k', zorder=0)   
        
    i = 0    
    zips = []    
    for x in LFPnorm:    
        zips.append(list(zip(t*x_zoom + electrode.x[i] + 2, x*z_zoom + electrode.z[i])))    
        i += 1    
        
    line_segments = LineCollection(zips, linewidths=(1.2), linestyles='solid', cmap='nipy_spectral', zorder=1)    
    line_segments.set_array(np.log10(-LFPmin))

    line_segments.set_norm(norm)    
    ax1.add_collection(line_segments)    
    
    axcb = fig.colorbar(line_segments, cax=cax, orientation='horizontal')  
    axcb.outline.set_visible(False)  

    valid_log_LFPmin = log_LFPmin[~np.isnan(log_LFPmin)]  
    min_val, max_val = np.min(valid_log_LFPmin), np.max(valid_log_LFPmin)  
    ticks = np.linspace(min_val, max_val, 5)  
    ticklabels = -10**ticks  
    axcb.set_ticks(ticks)  
    axcb.set_ticklabels(['{:.3f}'.format(t) for t in ticklabels])  
  
    axcb.set_label('spike amplitude (mV)', va='center')  
  
    ax1.set_xticks([])  
    ax1.set_yticks([])  
  
    axis = ax1.axis(ax1.axis('equal'))  
    ax1.set_xlim(axis[0]*1.02, axis[1]*1.02)  
  
    # plot morphology  
    zips = []  
    for x, z in cell.get_pt3d_polygons():  
        zips.append(list(zip(x, z)))  
  
    polycol = PolyCollection(zips, edgecolors='none', facecolors='gray', zorder=-1)  
    ax1.add_collection(polycol)  


def plot_morph_mea(cell, electrode):
    #color =  ['g', 'b', 'c', 'm', 'y', 'k']
    color = ['#88C0D0', '#5E81AC', '#BF616A', '#EBCB8B', '#A3BE8C', '#B48EAD']
    if not isinstance(cell, list):
        fig = plt.figure(dpi=160)
        ax1 = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax1.plot(electrode.x, electrode.z, 'o', markersize=1, color='r', zorder=0)
        zips = []
        for x, z in cell.get_pt3d_polygons():
            zips.append(list(zip(x, z)))
        polycol = PolyCollection(zips, edgecolors='none', facecolors='gray', zorder=-1, linewidths=0.5)
        ax1.add_collection(polycol)
        #ax1.set_xticks([])
        #ax1.set_yticks([])
        axis = ax1.axis(ax1.axis('equal'))
        ax1.set_xlim(axis[0]*1.02, axis[1]*1.02)
    else: 
        # plot multi-cell morphology
        fig = plt.figure(dpi=160)
        ax1 = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax1.plot(electrode.x, electrode.z, 'o', markersize=1, color='r', zorder=0)
        for i in range(len(cell)):
            polycol = PolyCollection(cell[i], edgecolors=color[i], zorder=-1, facecolors=color[i], linewidths=0.5)
            ax1.add_collection(polycol)
            #ax1.set_xticks([])
            #ax1.set_yticks([])
            axis = ax1.axis(ax1.axis('equal'))
            ax1.set_xlim(axis[0]*1.02, axis[1]*1.02)
    plt.savefig('morphology.eps', dpi=400)
    
def plot_morph_mea_3d(cell, electrode):
    fig = plt.figure(dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(electrode.x, electrode.y, electrode.z, marker='o', color='r', s=1, zorder=0)
    if not isinstance(cell, list):
        zips = []
        for values in cell.get_pt3d_polygons():
            if len(values) == 3: 
                x, y, z = values
                zips.append(list(zip(x, y, z)))
            elif len(values) == 2:
                x, z = values
                y = [0] * len(x) 
                zips.append(list(zip(x, y, z)))
            else:
                raise ValueError('Unexpected data format from cell.get_pt3d_polygons()')
        polycol = Poly3DCollection(zips, edgecolors='none', facecolors='gray', zorder=-1)
        ax.add_collection3d(polycol)
    else: 
        # ignore multi-cell
        pass

    #axis = ax.axis(ax.axis('equal'))
    #ax.set_xlim(axis[0]*1.02, axis[1]*1.02)
    ax.set_xlim(cell.x.min()*1.2, cell.x.max()*1.2)
    ax.set_ylim(cell.y.min()*1.2, cell.y.max()*1.2)
    ax.set_zlim(cell.z.min()*1.2, cell.z.max()*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()


