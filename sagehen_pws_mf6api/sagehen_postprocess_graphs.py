#!/usr/bin/env python
# coding: utf-8

# # MODFLOW API Paper
#
# ## Coupling of MODFLOW to PRMS
#
# This notebook can be used to reproduce published results for the "Coupling of MODFLOW to PRMS" example, as reported in the MODFLOW 6 API paper (in progress).
#
# ## Supported operating systems
# This example can be run on the following operating systems:
#
# * linux
# * macOS
# * Windows
#
# ## Prerequisites
# To process the results, the following publicly available software are required:
#
# * __flopy__ is a python package that can be used to build, run, and post-process MODFLOW 6 models. The source is available at https://github.com/modflowpy/flopy and the package can be installed from PyPI using `pip install flopy` or conda using `conda install flopy`.
# * __pandas__ which can be installed using PyPI (`pip install pandas`) or conda (`conda install pandas`).
# * __geopandas__ which can be installed using PyPI (`pip install geopandas`) or conda (`conda install geopandas`).
# * __fiona__ which can be installed using PyPI (`pip install fiona`) or conda (`conda install fiona`).
# * __netCDF4__ which can be installed using PyPI (`pip install netCDF4`) or conda (`conda install netCDF4`).
# * __hydrofunctions__ which can be installed using PyPI (`pip install hydrofunctions`) or conda (`conda install hydrofunctions`).
#
# ## Post-processing the results
#
# We start by importing the necessary packages:


import os
import pathlib as pl
import sys
import datetime
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import flopy
import geopandas as gpd

# import fiona
import netCDF4 as nc
import pandas as pd
import hydrofunctions as hf

get_ipython().run_line_magic("matplotlib", "inline")


# In[3]:


sys.path.append(os.path.join("../common"))
from figspecs import USGSFigure


fig_dir = pl.Path("figures")
fig_dir.mkdir(exist_ok=True)

# #### Figure dimensions and figure type

# In[4]:


figwidth = 85  # mm
figwidth = figwidth / 10 / 2.54  # inches

fs = USGSFigure(figure_type="graph")


# #### Process the geodatabase

# In[5]:

root_dir = pl.Path("../").resolve()
file = root_dir / "Sagehen.gdb"
hru = gpd.read_file(file, driver="FileGDB", layer="HRU")
river = gpd.read_file(file, driver="FileGDB", layer="stream")


# In[6]:


ws = root_dir / "sagehenmodel"
sim = flopy.mf6.MFSimulation().load(sim_ws=ws)
gwf = sim.get_model("sagehenmodel")


# ##### Set coordinate information for model grid

# In[7]:


gwf.modelgrid.set_coord_info(
    xoff=214860, yoff=4365620, epsg=26911, angrot=12.013768668935385975
)


# ##### Print model discretization shape

# In[8]:


gwf.modelgrid.shape, gwf.modelgrid.nnodes


# #### Get PRMS output from stand-alone run

# In[9]:


fpth = root_dir / "sagehenmodel/output/pywatershed_output.npz"
prms_out = np.load(fpth)


# #### Get prms output times

# In[10]:


idx0 = 0  # 365
times = prms_out["time"]
ndays = times.shape[0]
print("Number of PRMS days to process {}".format(ndays))


# #### Get MODFLOW output times

# In[11]:


tobj = flopy.utils.CellBudgetFile(
    root_dir / "sagehenmodel/output/gwf_sagehen-gsf.sfr.cbc", precision="double"
)
times = np.array(tobj.get_times())
ndays = times.shape[0]
print("Number of MODFLOW days to process {}".format(ndays))


# ##### Calculate cell area and conversion factors

# In[12]:


cell_area = 90.0 * 90.0
cum2m = 1.0 / cell_area
m2mm = 1000.0
cum2mpd = cum2m / ndays
cum2mmpd = cum2mpd * m2mm

m2ft = 3.28081
in2m = 1.0 / (12.0 * m2ft)

d2sec = 60.0 * 60.0 * 24.0


# #### Get observed streamflow data

# In[13]:


start = "1980-10-01"
start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
end_dt = start_dt + datetime.timedelta(days=ndays - 1)
end = end_dt.strftime("%Y-%m-%d")
start_dt, end_dt, start, end


# In[14]:


site = "10343500"
site_name = "Site {}".format(site)
sagehen = hf.NWIS(site, "dv", start, end)
sagehen


# In[15]:


sagehenStreamFlow = (sagehen.df()["USGS:10343500:00060:00003"] / (m2ft**3)).to_frame()
sagehenStreamFlow.rename(columns={"USGS:10343500:00060:00003": site_name}, inplace=True)


# #### Get simulated stream flow

# In[16]:


sobj = flopy.utils.CellBudgetFile(
    root_dir / "sagehenmodel/output/gwf_sagehen-gsf.sfr.cbc", precision="double"
)
sagehenSimQ_lst = []
for idx, totim in enumerate(times):
    sagehenSimQ_lst.append(
        -sobj.get_data(totim=totim, text="EXT-OUTFLOW")[0]["q"][-1] / d2sec
    )


# In[17]:


sagehenSimQ = pd.DataFrame(
    sagehenSimQ_lst, index=sagehenStreamFlow.index, columns=("Simulated",)
)
sagehenSimQ


# ##### Add simulate streamflow to dataframe

# In[18]:


sagehenStreamFlow["Simulated"] = sagehenSimQ["Simulated"]


# #### Set the plot times

# In[19]:


plt_times = sagehenStreamFlow.index[idx0:]

plt_times.shape


# #### Create streamflow figure

# In[20]:


def streamflow_fig():
    figheight = figwidth * 0.5
    fig, ax = plt.subplots(
        figsize=(figwidth, figheight),
        ncols=1,
        nrows=1,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(
        w_pad=4 / 72,
        h_pad=4 / 72,
        hspace=0,
        wspace=0,
    )

    for name, color in zip(
        (
            site_name,
            "Simulated",
        ),
        (
            "black",
            "red",
        ),
    ):
        ax.plot(
            plt_times,
            sagehenStreamFlow[name][idx0:],
            linewidth=0.25,
            color=color,
            label=name,
        )
    leg = fs.graph_legend(ax)

    ax.set_xlim(plt_times[0], plt_times[-1])
    # ax.set_ylim(0, 10)

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Streamflow (m$^3$/s)")

    fpth = pl.Path(root_dir / "figures/sagehen_pywatershed_streamflow.png")
    plt.savefig(fpth, dpi=600)


# ##### Save dataframe index

# In[21]:


df_index = sagehenStreamFlow.index


# ##### Get idomain for mapping

# In[22]:


idomain = gwf.dis.idomain.array


# #### Get HRU areas and convert to square meters

# In[23]:


ds = nc.Dataset(root_dir / "prms_grid_v3-Copy1.nc")
hru_area = ds["hru_area"][:]  # m2
acre2m2 = 43560.0 / (m2ft * m2ft)
hru_area *= acre2m2


# ##### Calculate model area

# In[24]:


nactive_cells = idomain[0, :, :].sum()
active_area = cell_area * nactive_cells
nactive_cells, active_area


# #### Process SFR, CBC, UZF budget output

# In[25]:


fpth = root_dir / "sagehenmodel/output/gwf_sagehen-gsf.sfr.cbc"
sfrobj = flopy.utils.CellBudgetFile(fpth, precision="double")
sfrobj.get_unique_record_names(0)


# In[26]:


cbcobj = flopy.utils.CellBudgetFile(
    root_dir / "sagehenmodel/output/gwf_sagehen-gsf.cbc", precision="double"
)
cbcobj.get_unique_record_names()


# In[27]:


uzfobj = flopy.utils.CellBudgetFile(
    root_dir / "sagehenmodel/output/gwf_sagehen-gsf.uzf.cbc", precision="double"
)
uzfobj.get_unique_record_names()


# #### Map PRMS data to MODFLOW grid

# In[28]:


list(prms_out.keys())


# In[29]:


prms_out["ppt"].shape


# #### Function to sum MODFLOW 6 terms

# In[30]:


def sum_terms(bobj, text="UZET", vmult=1.0, gridBased=False):
    v = np.zeros(times.shape[0], dtype=float)
    for idx, totim in enumerate(times):
        if gridBased:
            v[idx] = vmult * bobj.get_data(totim=totim, text=text)[0].sum()
        else:
            v[idx] = vmult * bobj.get_data(totim=totim, text=text)[0]["q"].sum()
    return v


# ##### Create empty data frame for summation arrays

# In[31]:


dfObj = pd.DataFrame(
    columns=(
        "ppt",
        "prms_actet",
        "uzf_actet",
        "gwf_actet",
        "prms_infil",
        "runoff",
        "interflow",
        "gwf_sto",
        "uzf_sto",
        "tot_sto",
        "underflow",
        "sfr_runoff",
        "seepage",
        "baseflow",
    ),
    index=df_index,
)


# #### Add PRMS flows to the data frame

# In[32]:


dfObj["ppt"] = np.sum(prms_out["ppt"][:ndays], axis=1) / active_area
dfObj["prms_actet"] = np.sum(prms_out["actet"][:ndays], axis=1) / active_area
dfObj["prms_infil"] = np.sum(prms_out["infil"][:ndays], axis=1) / active_area
dfObj["runoff"] = np.sum(prms_out["runoff"][:ndays], axis=1) / active_area
dfObj["interflow"] = np.sum(prms_out["interflow"][:ndays], axis=1) / active_area


# #### Add evapotranspiration flows to the data frame

# In[33]:


dfObj["uzf_actet"] = sum_terms(uzfobj, text="UZET", vmult=-1) / active_area
dfObj["gwf_actet"] = sum_terms(cbcobj, text="UZF-GWET", vmult=-1) / active_area


# #### Add storage flows to the data frame

# In[34]:


dfObj["uzf_sto"] = sum_terms(uzfobj, text="STORAGE") / active_area
dfObj["gwf_sto"] = sum_terms(cbcobj, text="STO-SS", gridBased=True) / active_area
dfObj["gwf_sto"] += sum_terms(cbcobj, text="STO-SY", gridBased=True) / active_area
dfObj["tot_sto"] = dfObj["uzf_sto"] + dfObj["gwf_sto"]


# #### Add streamflows to the data frame

# In[35]:


dfObj["baseflow"] = sum_terms(sfrobj, text="GWF") / active_area
dfObj["sfr_runoff"] = sum_terms(sfrobj, text="RUNOFF") / active_area
dfObj["interflow"] += sum_terms(uzfobj, text="REJ-INF-TO-MVR", vmult=-1.0) / active_area
dfObj["seepage"] = sum_terms(cbcobj, text="DRN-TO-MVR", vmult=-1.0) / active_area
dfObj["underflow"] = sum_terms(cbcobj, text="CHD", vmult=-1.0) / active_area


# ##### Function to calculate cumulative values

# In[36]:


def cum_calc(v, i0=idx0):
    return v[i0:].cumsum()


def et_recharge_ppt_fig():
    # #### Plot evapotranspiration terms with soil infiltration and precipitation
    vtot = np.zeros(plt_times.shape, dtype=float)
    colors = ("#c36f31", "#cab39f", "#b7bf5e")
    for name, color in zip(dfObj.columns[1:4], colors):
        v = cum_calc(dfObj[name])
        plt.fill_between(plt_times, vtot + v, y2=vtot, color=color, label=name)
        vtot += v
    # cum_et = vtot.copy()
    plt.plot(plt_times, cum_calc(dfObj["ppt"]), lw=1, color="cyan", label="ppt")
    plt.plot(
        plt_times,
        cum_calc(dfObj["prms_infil"]),
        lw=1,
        color="green",
        label="soil recharge",
    )
    plt.legend(loc="upper left")

    print(
        "total Rainfall {:.4g}".format(cum_calc(dfObj["ppt"])[-1]),
        "total ET {:.4g}".format(vtot[-1]),
        "prms_actet {:.4g} ({:.4%})".format(
            cum_calc(dfObj["prms_actet"])[-1],
            (cum_calc(dfObj["prms_actet"]) / vtot)[-1],
        ),
        "uzf_actet {:.4g} ({:.4%})".format(
            cum_calc(dfObj["uzf_actet"])[-1], (cum_calc(dfObj["uzf_actet"]) / vtot)[-1]
        ),
        "gwf_actet {:.4g} ({:.4%})".format(
            cum_calc(dfObj["gwf_actet"])[-1], (cum_calc(dfObj["gwf_actet"]) / vtot)[-1]
        ),
    )
    return


def gwf_uzf_storage_changes_fig():
    # plt.plot(df_index, dfObj["uzf_sto"], color="red")
    # plt.plot(df_index, dfObj["gwf_sto"], color="blue")
    # plt.plot(df_index, dfObj["tot_sto"], color="black")

    vtot = np.zeros(plt_times.shape, dtype=float)
    colors = ("#c36f31", "#cab39f", "#b7bf5e")
    for name, color in zip(
        (
            "uzf_sto",
            "gwf_sto",
        ),
        colors,
    ):
        v = cum_calc(dfObj[name])
        plt.fill_between(plt_times, vtot + v, y2=vtot, color=color, label=name)
        vtot += v
    plt.legend(loc="upper left")
    return


def cumulative_streamflow_fig():
    vtot = np.zeros(plt_times.shape, dtype=float)
    colors = (
        "#FF9AA2",
        "#FFB7B2",
        "#FFDAC1",
        "#E2F0CB",
        "#B5EAD7",
        "#C7CEEA",
    )[::-1]
    for name, color in zip(
        (
            "underflow",
            "runoff",
            "interflow",
            "seepage",
            "baseflow",
            "underflow",
        ),
        colors,
    ):
        v = cum_calc(dfObj[name])
        plt.fill_between(plt_times, vtot + v, y2=vtot, color=color, label=name)
        vtot += v
        # print(vtot[-1])

    plt.legend(loc="upper left")
    return


vtot = cum_calc(dfObj["runoff"])[-1]
vtot += cum_calc(dfObj["interflow"])[-1]
vtot += cum_calc(dfObj["baseflow"])[-1]
vtot += cum_calc(dfObj["seepage"])[-1]
print(
    " total observed streamflow {:.4g}\n".format(
        cum_calc(sagehenStreamFlow[site_name])[-1] * d2sec / active_area
    ),
    "total simulated streamflow {:.4g}\n".format(vtot),
    "runoff {:.4g} ({:.4%})".format(
        cum_calc(dfObj["runoff"])[-1], cum_calc(dfObj["runoff"])[-1] / vtot
    ),
    "interflow {:.4g} ({:.4%})".format(
        cum_calc(dfObj["interflow"])[-1], cum_calc(dfObj["interflow"])[-1] / vtot
    ),
    "baseflow {:.4g} ({:.4%})".format(
        cum_calc(dfObj["baseflow"])[-1], cum_calc(dfObj["baseflow"])[-1] / vtot
    ),
    "seepage {:.4g} ({:.4%})".format(
        cum_calc(dfObj["seepage"])[-1], cum_calc(dfObj["seepage"])[-1] / vtot
    ),
)


Qsim = sagehenStreamFlow["Simulated"][idx0:]
Qobs = sagehenStreamFlow[site_name][idx0:]
me = (Qsim - Qobs).mean()
Qmean = Qobs.mean()
numer = ((Qsim - Qobs) ** 2).sum()
denom = ((Qsim - Qmean) ** 2).sum()
nse = 1 - numer / denom
me, nse


def composite_fig():
    figheight = figwidth * 1.25
    fig, axes = plt.subplots(
        figsize=(figwidth, figheight),
        ncols=1,
        nrows=3,
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(
        w_pad=4 / 72,
        h_pad=4 / 72,
        hspace=0,
        wspace=0,
    )

    handletextpad = 0.1
    markerscale = 1
    handlelength = 1.25
    columnspacing = 0.5
    labelspacing = 0.25

    for idx, ax in enumerate(axes):
        ax.set_xlim(plt_times[0], plt_times[-1])
        if idx == 0:
            ax.set_ylabel(r"Streamflow (m$^3$/s)")
        else:
            ax.set_ylabel("Cumulative flow (m)")
        ax.get_yaxis().set_label_coords(-0.05, 0.5)

    ax = axes[0]
    ax.set_ylim(0, 0.35)
    zorder = 100
    for name, color, linestyle in zip(
        (
            site_name,
            "Simulated",
        ),
        (
            "blue",
            "black",
        ),
        (
            ":",
            "-",
        ),
    ):
        ax.plot(
            plt_times,
            sagehenStreamFlow[name][idx0:],
            linewidth=0.75,
            linestyle=linestyle,
            color=color,
            zorder=zorder,
            label=name,
        )
        zorder -= 10
    # ax.set_ylim(0, ax.get_ylim()[1])
    leg = fs.graph_legend(
        ax,
        ncol=2,
        loc="upper right",
        handletextpad=handletextpad,
        handlelength=handlelength,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
    )
    fs.heading(ax=ax, idx=0)
    fs.remove_edge_ticks()

    ax = axes[1]
    ax.set_ylim(0, 0.9)
    vtot = np.zeros(plt_times.shape, dtype=float)
    colors = (
        "#FF9AA2",
        "#FFB7B2",
        "#FFDAC1",
        "#E2F0CB",
        "#B5EAD7",
        "#C7CEEA",
    )[::-1]
    labels = (
        "Runoff",
        "Interflow",
        "Groundwater\nSeepage",
        "Baseflow",
        "Basin\nUnderflow",
    )
    names = (
        "runoff",
        "interflow",
        "seepage",
        "baseflow",
    )
    for name, color, label in zip(names, colors, labels):
        v = cum_calc(dfObj[name])
        ax.fill_between(plt_times, vtot + v, y2=vtot, color=color)
        ax.plot(
            [-1],
            [-1],
            lw=0,
            marker="s",
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        vtot += v
    # ax.set_ylim(0, ax.get_ylim()[1])
    fs.graph_legend(
        ax=ax,
        loc="upper left",
        ncol=2,
        handletextpad=handletextpad,
        markerscale=markerscale,
        handlelength=handlelength,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
    )

    ax = axes[2]
    ax.set_ylim(0, 0.9)
    vtot = np.zeros(plt_times.shape, dtype=float)
    colors_met = (
        "#FF6962",
        "#FFE08E",
        "#FFB346",
    )
    names_met = (
        "prms_actet",
        "uzf_actet",
        "gwf_actet",
    )
    labels_met = (
        "PRMS ET",
        "Unsaturated zone ET",
        "Groundwater ET",
    )

    # ax.plot(plt_times, cum_calc(dfObj["ppt"]), lw=1.25, color="cyan", label="Rainfall")
    for name, color, label in zip(names_met, colors_met, labels_met):
        v = cum_calc(dfObj[name])
        ax.fill_between(plt_times, vtot + v, y2=vtot, color=color)
        ax.plot(
            [-1],
            [-1],
            lw=0,
            marker="s",
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        vtot += v
    # ax.set_ylim(0, ax.get_ylim()[1])
    fs.graph_legend(
        ax=ax,
        loc="upper left",
        ncol=1,
        handletextpad=handletextpad,
        markerscale=markerscale,
        handlelength=handlelength,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
    )
    fs.heading(ax=ax, idx=1)
    fs.remove_edge_ticks()

    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%Y"))
    ax.xaxis.set_tick_params(rotation=45)
    fs.heading(ax=ax, idx=2)
    fs.remove_edge_ticks()

    fpth = pl.Path(root_dir / "figures/sagehen_pywatershed_graphs.png")
    plt.savefig(fpth, dpi=600)
    return
