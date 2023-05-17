import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

class time_conversion:
    def __init__(self):
        self.factor = {"min":1/60,"hr":1/(60*60),"day":1/(24*60*60)}

def markers_plot(markers_dataframe,save_in_path,pdf=False,time_unit="s"):

    print("Plotting marker displacements...")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    times = list(markers_dataframe.t.unique())
    if time_unit != "s":
        times = [time_conversion().factor[time_unit]*t for t in times]
    markers = markers_dataframe.label.unique()
    for marker in markers:
        df = markers_dataframe[markers_dataframe.label==marker].copy()
        df[["dx","dy"]] = df[["dx","dy"]]*100 #in cm
        df["D"] = np.sqrt(df["dx"]**2+df["dy"]**2)
        p = ax.plot(times,df["D"].to_list(),label=marker)
        ax.plot(times,df["dx"].to_list(),c=p[0].get_color(),ls="--")
    if len(markers) > 1:
        ax.legend()
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("$t$ ["+time_unit+"]")
    ax.set_ylabel("$D_g$ [cm]")
    ax.set_title("Marker displacements")
    filetype = ".pdf" if pdf else ".png"
    plt.savefig(os.path.join(save_in_path,"markers_plot"+filetype))
    plt.close()

def columns_plot(columns_dataframe,save_in_path,pdf=False,time_unit="s"):

    print("Plotting column displacements...")
    columns = columns_dataframe.label.unique()
    for column in columns:
        df = columns_dataframe[columns_dataframe.label == column].copy()

        #COLUMN PLOT
        fig = plt.figure()
        ax = fig.add_subplot(111)
        times = df["t"].unique()
        every = int(len(times)/10) if len(times) >= 20 else 1
        a = dict(zip(times[::every],np.linspace(0.1,1,len(times[::every]))))
        y0 = df[(df.label == column) & (df.t==0)][["node","y0"]].set_index("node")
        for t in times[::every]:
            dx = df[df["t"]==t][["node","dx"]].set_index("node")
            dy = df[df["t"]==t][["node","dy"]].set_index("node")
            y = dy.join(y0)
            y["y"] = y["y0"]+y["dy"]
            ax.plot(dx,list(y["y"]),"-o",alpha=a[t],color="b")

        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$\Delta z$ [m]")
        ax.set_ylabel(r"$x$ [m]")
        ax.set_title("Column displacements: "+column)
        filetype = ".pdf" if pdf else ".png"
        plt.savefig(os.path.join(save_in_path,column+"_column_plot"+filetype))
        plt.close()

        #change time unit
        if time_unit != "s":
            df["t"] = df["t"].apply(lambda t: t*time_conversion().factor[time_unit])
            times = df["t"].unique()

        #NODAL DISPLACEMENTS PLOT
        fig,axs = plt.subplots(2,sharex=True)
        ax1, ax2 = axs[0], axs[1]

        nodes = list(df["node"].unique())
        nodes.reverse()
        colors = plt.cm.jet(np.linspace(0,1,len(nodes)))
        for node in nodes:
            dx = [(x)*100 for x in np.array(df[(df["node"]==node)]["dx"])]
            dy = [(y)*100 for y in np.array(df[(df["node"]==node)]["dy"])]
            ax1.plot(times,dx,color=colors[nodes.index(node)])
            ax2.plot(times,dy,color=colors[nodes.index(node)])

        ax1.set_title("Nodal displacements: "+column)
        ax1 = plot_aux(ax1,ylabel=r"$\Delta z$ [cm]")
        ax2 = plot_aux(ax2,xlabel="$t$ ["+time_unit+"]",ylabel=r"$\Delta x$ [cm]")
        ax1.yaxis.tick_right()
        #ax1.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        #ax2.yaxis.set_label_position("right")

        cmap = plt.get_cmap('jet',len(nodes))
        norm = mpl.colors.Normalize(vmin=0, vmax=len(nodes))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[:],ticks=list(range(len(nodes))),pad=0.1)
        cbar.ax.set_yticklabels([str(n) for n in nodes])
        cbar.ax.set_title("Node")

        filetype = ".pdf" if pdf else ".png"
        plt.savefig(os.path.join(save_in_path,column+"_nodal_displacements"+filetype))
        plt.close()

def precipitation_plot(precipitation_dataframe,save_in_path,gwt=False,pdf=False,time_unit="s"):

    print("Plotting precipitation ...")
    fig = plt.figure()
    ax = fig.add_subplot(111) if not gwt else fig.add_subplot(211)
    df = precipitation_dataframe
    cumulative = sum(df.value)
    one_day = 24*60*60
    if time_unit != "s":
        factor = time_conversion().factor[time_unit]
        one_day = one_day*factor
        df.t = df.t.apply(lambda t: t*factor)

    width = df.t.iloc[1]-df.t.iloc[0]
    ax.bar(list(df.t),list(df.value),width=width,color="tab:blue")

    df["cml_1"] = df.apply(lambda x: df.loc[(df.t <= x.t) & (df.t > x.t-24), 'value'].sum(),axis=1)
    axt = ax.twinx()
    axt.plot(list(df.t),list(df.cml_1),c="tab:red")

    if not gwt:
        ax.set_xlabel("$t$ ["+time_unit+"]")
    ax.set_ylabel(r"$P$ [mm]",c="tab:blue")
    axt.set_ylabel(r"$P_c$ [mm]",c="tab:red")
    ax.set_axisbelow(True)
    ax.grid(True)

    if gwt:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax2 = fig.add_subplot(212,sharex=ax)
        ax2.plot(list(precipitation_dataframe.t),list(precipitation_dataframe.gwt_depth_change))
        ax2.set_xlabel("$t$ ["+time_unit+"]")
        ax2.set_ylabel(r"$\Delta x_{gw}$ [m]")
        ax2.set_axisbelow(True)
        ax2.grid(True)

    filetype = ".pdf" if pdf else ".png"
    plt.savefig(os.path.join(save_in_path,"precipitation_plot"+filetype))
    plt.close()

def factor_of_safety_at_step(slip_surfaces_labels_to_params,slip_surfaces_dataframe,print_label=False):

    Fs_dict = {}
    slip_surfaces = slip_surfaces_dataframe.label.unique()
    for slip_surface in slip_surfaces:
        if print_label:
            print(slip_surface)
        df = slip_surfaces_dataframe[slip_surfaces_dataframe.label==slip_surface].copy()
        df.friction_angle = df.friction_angle.apply(lambda x: np.tan(x))
        df["normal_stress_eff"] = df.normal_stress-df.pore_pressure
        df = df.drop(["normal_stress","pore_pressure"],axis=1)
        df["cohesion"] = df[["slice","cohesion"]].groupby("slice")["cohesion"].transform("first")
        df["friction_angle"] = df[["slice","friction_angle"]].groupby("slice")["friction_angle"].transform("first")
        df.loc[df.normal_stress_eff < 0,"shear_strength"] = df.cohesion - df.normal_stress_eff*df.friction_angle
        df["shear_strength"] = df["shear_strength"]*np.sign(df["shear_stress"])
        
        angles = slip_surfaces_labels_to_params[slip_surface].angles
        df["angle"] = angles
        df = df[df.shear_strength.notnull()]

        width = slip_surfaces_labels_to_params[slip_surface].arc_params[2]
        df["moment"] = width*df.shear_stress/np.cos(df.angle)
        df["max_moment"] = width*df.shear_strength/np.cos(df.angle)
        Fs_dict.update({slip_surface:df.max_moment.sum()/df.moment.sum()})

    return pd.DataFrame.from_dict({"label":list(Fs_dict.keys()),"Fs":list(Fs_dict.values())})

def factor_of_safety_evolution(slip_surfaces_labels_to_params,slip_surfaces_dataframe):

    df = slip_surfaces_dataframe
    df["cohesion"] = df[["slice","cohesion"]].groupby("slice")["cohesion"].transform("first")
    df["friction_angle"] = df[["slice","friction_angle"]].groupby("slice")["friction_angle"].transform("first")
    df = df.drop(["x0","y0"],axis=1)

    Fs_df_out = pd.DataFrame(columns=["label","t","Fs"])
    times = df.t.unique()
    for t in times:
        Fs_df = factor_of_safety_at_step(slip_surfaces_labels_to_params,df[df.t==t])
        Fs_df["t"] = t
        Fs_df_out = pd.concat([Fs_df_out,Fs_df],ignore_index=True)
    return Fs_df_out

def factor_of_safety_plot(slip_surfaces_labels_to_params,slip_surfaces_dataframe,save_in_path,ylim_max=None,pdf=False,time_unit="s"):

    print("Plotting factor-of-safety of slip surfaces ...")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    Fs_df = factor_of_safety_evolution(slip_surfaces_labels_to_params,slip_surfaces_dataframe)
    if time_unit != "s":
        Fs_df["t"] = Fs_df["t"].apply(lambda t: t*time_conversion().factor[time_unit])

    slip_surfaces = Fs_df.label.unique()
    colors = plt.cm.jet(np.linspace(0,1,len(slip_surfaces)))
    i = 0
    for slip_surface in slip_surfaces:
        df = Fs_df[Fs_df.label == slip_surface]
        negative_Fs = list(df[df.Fs < 0].t)
        if len(negative_Fs) > 0:
            t_cut = negative_Fs[0]
            ax.plot(list(df[df.t <= t_cut].t),list(df[df.t <= t_cut].Fs),marker=".",label=slip_surface,c=colors[i])
        elif len(negative_Fs) == 0:
            ax.plot(list(df.t),list(df.Fs),marker=".",label=slip_surface,c=colors[i])
        i += 1

    if ylim_max is not None:
        ax.set_ylim(0,ylim_max)
    else:
        ax.set_ylim(0)
    ax.axhline(1,c="k",ls=":")
    if len(slip_surfaces) == 1:
        ax.set_title("Factor-of-Safety of slip surface: "+slip_surface)
    else:
        ax.set_title("Factor-of-Safety of slip surfaces")
    #legend = True if len(slip_surface) > 1 else False
    legend = True
    ax = plot_aux(ax,xlabel=r"$t$ ["+time_unit+"]",ylabel=r"$F_S$",legend=legend)
    filetype = ".pdf" if pdf else ".png"
    plt.savefig(os.path.join(save_in_path,"factor_of_safety_plot"+filetype))
    plt.close()

    Fs_df = Fs_df.sort_values(["label","t"],ignore_index=True)
    Fs_df.to_csv(os.path.join(save_in_path,"factor_of_safety_data.csv"))

def slip_surfaces_stresses_plot(slip_surfaces_dataframe,save_in_path,pdf=False,time_unit="s"):

    print("Plotting stresses in slip surfaces ...")
    df = slip_surfaces_dataframe
    df = df.drop(["x0","y0"],axis=1)
    times = list(df.t.unique())
    if time_unit != "s":
        times = [t*time_conversion().factor[time_unit] for t in times]

    slip_surface = df.label.unique()
    for stress_surface in slip_surface:
        fig, axs = plt.subplots(4,sharex=True)
        ax1, ax2, ax3, ax4 = axs[0], axs[1], axs[2], axs[3]

        df_ = df[df.label==stress_surface].copy()
        slices = list(df_.slice.unique())
        colors = plt.cm.jet(np.linspace(0,1,len(slices)))
        for slice in slices:
            df_s = df_[df_.slice==slice].copy()
            cohesion = df_s[df_s.t == 0]["cohesion"].iloc[0]
            friction_angle = df_s[df_s.t == 0]["friction_angle"].iloc[0]
            df_s["normal_stress_eff"] = (df_s.normal_stress-df_s.pore_pressure)
            df_s.loc[df_s.normal_stress_eff < 0, "shear_strength"] = cohesion-df_s.normal_stress_eff*np.tan(friction_angle)
            df_s["shear_strength"] = df_s["shear_strength"]*np.sign(df_s["shear_stress"])
            ax1.plot(times,list(df_s.normal_stress/1000),c=colors[slice])
            ax2.plot(times,list(df_s.pore_pressure/1000),c=colors[slice])
            ax3.plot(times,list(df_s.shear_stress/1000),c=colors[slice])
            ax4.plot(times,list(df_s.shear_strength/1000),c=colors[slice])

        ax_list = [ax1, ax2, ax3, ax4] #< your axes objects
        ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
        ax1.set_title("Stress at slip surface: "+stress_surface)
        ax1 = plot_aux(ax1,ylabel=r"$\sigma_n$ [kPa]")
        ax2 = plot_aux(ax2,ylabel=r"$p$ [kPa]")
        ax3 = plot_aux(ax3,ylabel=r"$\tau$ [kPa]")
        ax4 = plot_aux(ax4,xlabel=r"$t$ ["+time_unit+"]",ylabel=r"$\tau_{MC}$ [kPa]")

        cmap = plt.get_cmap('jet',len(slices))
        norm = mpl.colors.Normalize(vmin=0, vmax=len(slices))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[:], ticks=[0,max(slices)])
        cbar.ax.set_yticklabels(['leftmost', 'rightmost'],rotation=45,verticalalignment="center")
        cbar.ax.set_title("Slice")

        filetype = ".pdf" if pdf else ".png"
        plt.savefig(os.path.join(save_in_path,stress_surface+"_stress_plot"+filetype))
        plt.close()

def plot_aux(ax,xlabel="",ylabel="",legend=False):
    ax.grid()
    ax.set_axisbelow(True)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    if xlabel != "":
        ax.set_xlabel(xlabel)
    if legend:
        ax.legend(fontsize=8)
    return ax

def factor_of_safety(stress_surface,stresses,pore_pressures,cohesions,friction_angles):
    normal_stresses = []
    shear_stresses = []
    shear_strengths = []
    for i in range(len(stresses)):
        S = np.reshape(stresses[i],(2,2))
        S = S - pore_pressures[i]*np.eye(2)
        a = stress_surface.angles[i]
        R = np.reshape([np.cos(a),np.sin(a),-np.sin(a),np.cos(a)],(2,2))
        S = R*S*R.T
        normal_eff = (S[1][1]-pore_pressures[i][0])
        shear_stress = S[0][1]
        shear_strength = cohesions[i][0]+normal_eff*np.tan(friction_angles[i][0])
        if normal_eff < 0:
            normal_eff = 0.
            shear_stress = 0.
            shear_strength = 0.
        normal_stresses.append(normal_eff)
        shear_stresses.append(shear_stress)
        shear_strength = -shear_strength if shear_stress < 0 else shear_strength
        shear_strengths.append(shear_strength)

    fos = sum(shear_strengths)/sum(shear_stresses)

    return fos, normal_stresses, shear_stresses, shear_strengths
