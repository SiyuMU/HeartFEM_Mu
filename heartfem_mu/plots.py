"""Figures and movies (matplotlib; the movies also vtk and ffmpeg). One function per pipeline step that makes figures:

    pv_figures        volume_constrain, step plot: pv_loop.png/.pdf (300 dpi) from summary.csv and pv_cycle.mp4
    windkessel_figures(group)  windkessel, steps figures_table / figures_cycle / figures_fields:
        table   pvt_table.png             the P-V-t table the circuit reads: 3-D views, the working range, and a map
                                          (PINN: hatched outside the trained pressure range of the network)
        cycle   pv_loops.png              P-V loops of the atrium and the ventricle, with the valve events
                flows.png                 flows through the inlet, the two valves and the systemic resistance
                pressures_volumes.png     pressures and chamber volumes over the cycle
                metrics.csv               metrics of the last cycle next to the measured values
        fields  replay/fields_summary.*   the fields of the coupled cycle per state (csv) and in four lines (txt):
                                          displacement, sarcomere length, J, wall-mean stresses next to a Laplace estimate
                pv_cycle.mp4              the coupled cycle state by state: the deforming wall on a long-axis section
                                          coloured by the fiber stress, next to the atrial P-V loop
    invert_comparison invert_para, step evaluate: comparison.png (long-axis sections, tracking vs model)
The colour of the solver: FE blue, PINN orange (volume_constrain).
"""
import csv
import json
import shutil

import numpy as np

from . import model, windkessel

BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#52514e", "#1b9e77"
MMHG = model.MMHG_PER_KPA


def long_axis_section(cfg, X, cells):
    """(section(x, values), o, e1, n): long-axis section through the apex of the wall mesh (vertices X um, tetrahedra
    cells) at deformed positions x -> polygons (in-plane um, base at 0) and cell values; o, e1, n: base origin, in-plane
    direction towards the apex offset, long axis (base -> apex)."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    lm = json.loads((cfg.input / "base_plane.json").read_text())
    o, n, apex = (np.asarray(lm[k], float) for k in ("origin", "normal", "apex"))
    n /= np.linalg.norm(n)
    n *= np.sign((apex - o) @ n)  # long axis, base -> apex
    e1 = (apex - o) - ((apex - o) @ n) * n
    e1 /= np.linalg.norm(e1)  # in-plane direction towards the apex offset
    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    grid.SetPoints(pts)
    ca = vtk.vtkCellArray()
    ca.SetCells(len(cells), numpy_to_vtkIdTypeArray(np.c_[np.full(len(cells), 4), cells].astype(np.int64).ravel(), deep=True))
    grid.SetCells(vtk.VTK_TETRA, ca)
    plane = vtk.vtkPlane()
    plane.SetOrigin(*o)
    plane.SetNormal(*np.cross(n, e1))
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)

    def section(x, values):
        """Polygons (in-plane um, base at 0) and cell values of the long-axis section of the mesh at positions x."""
        pts.SetData(numpy_to_vtk(x, deep=True))
        arr = numpy_to_vtk(np.asarray(values, np.float64), deep=True)
        arr.SetName("v")
        grid.GetCellData().AddArray(arr)
        grid.Modified()
        cutter.Update()
        cut = cutter.GetOutput()
        xy = (vtk_to_numpy(cut.GetPoints().GetData()) - o) @ np.c_[e1, n]
        polys = [xy[[cut.GetCell(c).GetPointId(m) for m in range(cut.GetCell(c).GetNumberOfPoints())]]
                 for c in range(cut.GetNumberOfCells())]
        return polys, vtk_to_numpy(cut.GetCellData().GetArray("v"))
    return section, o, e1, n


def ffmpeg_writer(cfg, plt):
    """FFMpegWriter of the movies, or None (with a message) if ffmpeg (params.FFMPEG) is not found."""
    from matplotlib.animation import FFMpegWriter
    ffmpeg = shutil.which(cfg.FFMPEG)
    if ffmpeg is None:
        print(f"ffmpeg ({cfg.FFMPEG}) not found: no movie", flush=True)
        return None
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg
    return FFMpegWriter(fps=cfg.MOVIE_FPS, bitrate=4000)


# ==================== volume_constrain ====================
def pv_figures(cfg):
    """Step plot of volume_constrain: pv_loop.png/.pdf, then pv_cycle.mp4 (same process: the movie keeps the figure's
    rcParams)."""
    pv_loop(cfg)
    pv_movie(cfg)


def pv_loop(cfg):
    """OUTPUT_DIR/pv_loop.png (300 dpi) and pv_loop.pdf from summary.csv: (a) P-V loop, (b-d) P(t), V(t), T(t)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(cfg.output / "summary.csv")))
    k = np.array([int(r["frame"]) for r in rows])
    V, P, T, t_act = (np.array([float(r[c]) for r in rows]) for c in ("V_nL", "P_kPa", "T_mean_kPa", "t_active_ms"))
    ed, es, pk = int(np.argmax(V)), int(np.argmin(V)), int(np.argmax(P))
    t = np.mod(k - k[ed], cfg.N_FRAMES) * cfg.BCL_MS / cfg.N_FRAMES  # time since the maximum volume
    o = np.argsort(t)
    loop = np.r_[o, o[:1]]
    color, name = (BLUE, "FE") if cfg.SOLVER == "FE" else (ORANGE, "PINN")
    grey, shade = "#52514e", "#ececea"
    mmhg = 7.50062
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    fig = plt.figure(figsize=(7.2, 4.0), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.1, 1])
    a = left.subplots()
    b, c, d = right.subplots(3, 1, sharex=True)

    # (a) P-V loop: filled = active (wall-mean T > 1e-3 kPa), open = passive; arrow = direction of time.
    active = T > 1e-3
    a.plot(V[loop], P[loop], "-", color=color, zorder=1)
    a.scatter(V[active], P[active], s=14, color=color, zorder=2, label="active (T > 0)")
    a.scatter(V[~active], P[~active], s=14, facecolor="white", edgecolor=color, lw=0.8, zorder=2, label="passive (T = 0)")
    j = o[np.argmin(np.abs(P[o[:np.argmax(P[o])]] - 0.5 * P[pk]))]  # on the rising limb, half the peak pressure
    nxt = o[(np.flatnonzero(o == j)[0] + 1) % len(o)]
    a.annotate("", (V[nxt], P[nxt]), (V[j], P[j]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=12))
    lead = dict(arrowstyle="-", color=grey, lw=0.5, shrinkA=1, shrinkB=3)
    for i, lab, off, ha in ((pk, "peak P", (0, 8), "center"), (ed, "max V", (-18, 24), "right"),
                            (es, "min V", (22, 24), "left")):
        a.annotate(f"t{k[i]} {lab}, {P[i]:.2f} kPa", (V[i], P[i]), textcoords="offset points", xytext=off, ha=ha,
                   va="center", fontsize=7, color=grey, arrowprops=None if lab == "peak P" else lead)
    a.set(xlabel="cavity volume (nL)", ylabel="cavity pressure (kPa)", ylim=(-0.05, 1.12 * P.max()))
    a.secondary_yaxis("right", functions=(lambda x: x * mmhg, lambda x: x / mmhg)).set_ylabel("mmHg")
    a.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.42))
    a.set_title(f"(a) P-V loop ({name})", loc="left", fontweight="bold")

    # (b-d) time courses over one cycle from the maximum volume; grey band = active tension, dashed = minimum volume.
    dt = cfg.BCL_MS / cfg.N_FRAMES
    if cfg.SOLVER == "FE":  # activation onset, from this run's t_active
        on = np.mod(-t_act[ed], cfg.BCL_MS)
    else:
        on = np.mod(model.onset_frame(cfg) - k[ed], cfg.N_FRAMES) * dt
    end = t[active].max() + dt / 2 if active.any() else on  # last active frame (wall-mean T > 1e-3 kPa)
    for ax_, y, lab, title in ((b, P, "kPa", "(b) cavity pressure"), (c, V, "nL", "(c) cavity volume"),
                               (d, T, "kPa", "(d) wall-mean active tension")):
        ax_.axvspan(on, end, color=shade, lw=0, zorder=0)
        ax_.axvline(t[es], color=grey, lw=0.7, ls="--", zorder=1)
        ax_.plot(t[o], y[o], "-o", color=color, ms=2, zorder=2)
        ax_.set_ylabel(lab)
        ax_.set_title(title, loc="left", fontweight="bold")
    b.secondary_yaxis("right", functions=(lambda x: x * mmhg, lambda x: x / mmhg)).set_ylabel("mmHg")
    d.text(end, 0.95, " active", transform=d.get_xaxis_transform(), va="top", fontsize=7, color=grey)
    c.text(t[es], 0.95, f" min V (t{k[es]})", transform=c.get_xaxis_transform(), va="top", fontsize=7, color=grey)
    d.set_xlabel(f"time since maximum volume, t{k[ed]} (ms)")
    d.set_xlim(0, cfg.BCL_MS)
    for ax_ in (a, b, c, d):
        ax_.grid(alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.output / f"pv_loop.{ext}")
    plt.close(fig)
    print("wrote", cfg.output / "pv_loop.png", "and .pdf", flush=True)


def pv_movie(cfg):
    """OUTPUT_DIR/pv_cycle.mp4: left the deforming wall on a long-axis section through the apex (mesh lines, coloured
    by sigma_ff), right the P-V loop with the current frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    writer = ffmpeg_writer(cfg, plt)
    if writer is None:
        return
    r = np.load(cfg.output / "pv_results.npz")
    X, cells = r["points"] / cfg.MM_PER_UNIT, r["cells"]
    section, o, e1, n = long_axis_section(cfg, X, cells)
    sff = r["sigma_ff"]
    lim = float(np.ceil(np.percentile(np.abs(sff[:, r["far_from_base"]]), 99.5)))  # whole kPa: same scale in FE and PINN
    k, V, P, T = r["frame"], r["V_nL"], r["P_kPa"], r["T_mean_kPa"]
    ed = int(np.argmax(V))
    order = np.roll(np.arange(len(k)), -ed)  # one cycle from the maximum volume
    loop = np.r_[order, order[:1]]
    line, accent, grey = (BLUE, ORANGE, GREY) if cfg.SOLVER == "FE" else (ORANGE, BLUE, GREY)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    pc = PolyCollection([], cmap="RdBu_r", edgecolors="#00000026", linewidths=0.15)
    pc.set_clim(-lim, lim)
    a1.add_collection(pc)
    ext = np.concatenate([((X + 1e3 * r["u"][i]) - o) @ np.c_[e1, n] for i in range(len(k))])
    a1.set_xlim(ext[:, 0].min() - 5, ext[:, 0].max() + 5)
    a1.set_ylim(ext[:, 1].max() + 5, -5)  # base on top, apex at the bottom
    a1.set_aspect("equal")
    a1.set(xlabel="µm", ylabel="distance from the base plane (µm)")
    fig.colorbar(pc, ax=a1, shrink=0.8, extend="both", label="fiber stress σ$_{ff}$ (kPa)")
    a2.plot(V[loop], P[loop], "-", color=line, lw=2)
    a2.scatter(V, P, s=12, color=line)
    dot, = a2.plot([], [], "o", ms=12, color=accent, mec="white", mew=1.5)
    a2.set(xlabel="cavity volume (nL)", ylabel="cavity pressure (kPa)")
    a2.grid(alpha=0.3)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    label = a2.set_title("P-V loop\n ", loc="left", fontsize=10)  # two lines: the layout reserves room for them
    title = a1.set_title("", loc="left", fontsize=10)
    fig.tight_layout()
    dt = cfg.BCL_MS / cfg.N_FRAMES
    with writer.saving(fig, str(cfg.output / "pv_cycle.mp4"), dpi=130):
        for j, i in enumerate(order):
            polys, vals = section(X + 1e3 * r["u"][i], sff[i])
            pc.set_verts(polys)
            pc.set_array(vals)
            dot.set_data([V[i]], [P[i]])
            title.set_text(f"{cfg.SOLVER}, t{k[i]:02d}: long-axis section through the apex")
            label.set_text(f"P-V loop: {j * dt:.0f} ms after max volume\nV = {V[i]:.3f} nL,  P = {P[i]:.3f} kPa,  T = {T[i]:.1f} kPa")
            writer.grab_frame()
    print("wrote", cfg.output / "pv_cycle.mp4", flush=True)


# ==================== windkessel ====================
T_LABEL = "time from atrial activation onset (ms)"


def metric_rows(cfg, m):
    """(part, metric, value, measured). Measured: Salehin et al. 2021 (zebrafish 3 dpf); ventricular volumes:
    Cairelli et al. 2024 (3 dpf). Timing from the valve events (windkessel.valve_events)."""
    back = 100 * m["la_inflow_back"] / m["av_fwd"]
    return [
        (f"atrium ({cfg.SOLVER})", "volume max / min (nL)", f"{m['la_Vmax']:.3f} / {m['la_Vmin']:.3f}", "-"),
        ("", "ejection fraction", f"{100 * m['la_EF']:.0f} %", "-"),
        ("", "pressure max / min (mmHg)", f"{m['la_Pmax']:.2f} / {m['la_Pmin']:.2f}", "-"),
        ("", "a-wave (mmHg)", f"{m['la_a_wave']:.2f}", "-"),
        ("ventricle", "EDV / ESV (nL)", f"{m['lv_EDV']:.3f} / {m['lv_ESV']:.3f}", "0.57 / 0.23"),
        ("", "ejection fraction", f"{100 * (1 - m['lv_ESV'] / m['lv_EDV']):.0f} %", "55-60 %"),
        ("", "EDP / peak / ESP (mmHg)", f"{m['lv_EDP']:.2f} / {m['lv_Pmax']:.2f} / {m['lv_ESP']:.2f}", "1.36 / 7.52 / -"),
        ("", "filling / IVC / ejection / IVR (ms)",
         f"{m['lv_filling_ms']:.0f} / {m['lv_ivc_ms']:.0f} / {m['lv_ejection_ms']:.0f} / {m['lv_ivr_ms']:.0f}", "280 / 50 / 180 / 21"),
        ("", "end-diastole / end-systole (ms)", f"{m['lv_t_ED_ms']:.0f} / {m['lv_t_ES_ms']:.0f}", "-"),
        ("", "AV closure to contraction (ms)", f"{m['lv_wait_ms']:.0f}", "~0"),
        ("", "filling during atrial contraction", f"{100 * m['A_fraction']:.0f} %", "mainly atrial"),
        ("circulation", "arterial P min / mean / max (mmHg)", f"{m['art_Pmin']:.2f} / {m['art_Pmean']:.2f} / {m['art_Pmax']:.2f}", "mean 4.90"),
        ("", "venous P mean (mmHg)", f"{m['ven_Pmean']:.2f}", "-"),
        ("", "stroke volume (nL) / output (nL/s)", f"{m['out_fwd']:.3f} / {m['cardiac_output_nL_s']:.2f}", "-"),
        ("", "backflow atrium to veins (nL)", f"{m['la_inflow_back']:.3f} ({back:.0f} % of AV flow)", "-"),
        ("numerics", "periodicity, last two cycles (nL)", f"{m['periodic_err']:.1e}", "-"),
    ]


def phases(cfg, a, m, labels=True):
    """Vertical lines at the four valve events and, above the axes, the names of the ventricular phases."""
    ev = [m["lv_t_ED_ms"], m["lv_t_ejection_start_ms"], m["lv_t_ES_ms"], m["lv_t_AV_open_ms"]]
    for x in ev:
        a.axvline(x, color=GREY, lw=0.7, ls=":")
    if labels:
        bcl = cfg.BCL_MS
        for x0, x1, lab in ((0, ev[0], "filling by atrial contraction"), (ev[0], ev[1], "IVC"), (ev[1], ev[2], "ejection"),
                            (ev[2], ev[3], "IVR"), (ev[3], bcl, "passive filling")):
            a.text((x0 + x1) / 2, 1.01, lab, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=GREY)
    a.set_xlim(0, cfg.BCL_MS)


def pv_figure(cfg, plt, c, m):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    a = ax[0]
    a.plot(c["V_la"], c["P_la"], color=BLUE, lw=2.2)
    i_pk, i_lo = int(np.argmax(c["P_la"][:-1])), int(np.argmin(c["P_la"][:-1]))
    for i, lab, off in ((0, "activation onset", (-8, 10)), (i_pk, "peak pressure", (0, 8)),
                        (int(round(m["la_t_Vmin_ms"])), "minimum volume", (10, 0)), (i_lo, "minimum pressure", (6, -14))):
        a.plot(c["V_la"][i], c["P_la"][i], "o", color=BLUE, ms=6, mec="white", mew=1.2)
        a.annotate(f"{lab}, {c['t'][i] * 1e3:.0f} ms", (c["V_la"][i], c["P_la"][i]), textcoords="offset points", xytext=off,
                   fontsize=8, ha="right" if off[0] < 0 else ("center" if off[0] == 0 else "left"))
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="volume (nL)", ylabel="pressure (mmHg)")
    a.set_title("atrium (FE model, P-V-t table)" if cfg.SOLVER == "FE" else "atrium (PINN, P-V-t table)", loc="left")
    a.margins(0.12)
    a = ax[1]
    a.plot(c["V_lv"], c["P_lv"], color=ORANGE, lw=2.2)
    for key, lab, off, ha in (("lv_t_ED_ms", "AV valve closes (end-diastole)", (-10, 13), "right"),
                              ("lv_t_ejection_start_ms", "outflow valve opens", (-8, 0), "right"),
                              ("lv_t_ES_ms", "outflow valve closes (end-systole)", (8, 0), "left"),
                              ("lv_t_AV_open_ms", "AV valve opens", (6, -13), "left")):
        i = int(round(m[key])) % (len(c["t"]) - 1)
        a.plot(c["V_lv"][i], c["P_lv"][i], "o", color=ORANGE, ms=6, mec="white", mew=1.2)
        a.annotate(f"{lab}, {m[key]:.0f} ms", (c["V_lv"][i], c["P_lv"][i]), textcoords="offset points", xytext=off,
                   fontsize=8, ha=ha, va="center")
    a.set(xlabel="volume (nL)", ylabel="pressure (mmHg)")
    a.set_title("ventricle (time-varying elastance)", loc="left")
    a.margins(0.12)
    fig.savefig(cfg.output / "pv_loops.png", dpi=150)
    plt.close(fig)
    print("wrote", cfg.output / "pv_loops.png")


def flow_figure(cfg, plt, c, m, p):
    t = c["t"] * 1e3
    fig, a = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    q_sys = (c["P_art"] - c["P_ven"]) / p["R_sys"]
    a.plot(t, c["Q_venla"], color=GREEN, lw=1.8, label="veins -> atrium (no valve; negative = backflow)")
    a.plot(t, c["Q_av"], color=BLUE, lw=1.8, label="atrium -> ventricle (AV valve)")
    a.plot(t, c["Q_out"], color=ORANGE, lw=1.8, label="ventricle -> arteries (outflow valve)")
    a.plot(t, q_sys, color=GREY, lw=1.2, ls="--", label="arteries -> veins (systemic resistance)")
    a.axhline(0, color="k", lw=0.5)
    phases(cfg, a, m)
    a.set(xlabel=T_LABEL, ylabel="flow (nL/s)")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(cfg.output / "flows.png", dpi=150)
    plt.close(fig)
    print("wrote", cfg.output / "flows.png")


def pressure_volume_figure(cfg, plt, c, m):
    t = c["t"] * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    a = ax[0]
    for k, lab, col in (("P_la", "atrium", BLUE), ("P_lv", "ventricle", ORANGE), ("P_art", "arteries", GREY), ("P_ven", "veins", GREEN)):
        a.plot(t, c[k], color=col, lw=1.8, label=lab)
    a.axhline(0, color="k", lw=0.5)
    phases(cfg, a, m)
    a.set(ylabel="pressure (mmHg)")
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a = ax[1]
    a.plot(t, c["V_la"], color=BLUE, lw=1.8, label="atrium")
    a.plot(t, c["V_lv"], color=ORANGE, lw=1.8, label="ventricle")
    phases(cfg, a, m, labels=False)
    a.set(xlabel=T_LABEL, ylabel="volume (nL)")
    a.legend(fontsize=8, frameon=False, loc="upper center")
    fig.savefig(cfg.output / "pressures_volumes.png", dpi=150)
    plt.close(fig)
    print("wrote", cfg.output / "pressures_volumes.png")


def table_figure(cfg, plt):
    from matplotlib.colors import SymLogNorm
    V, t_ms, P = windkessel.read_total_table(cfg.output / "table")  # P[t, V], mmHg
    fig = plt.figure(figsize=(21, 13))
    TT, VV = np.meshgrid(t_ms, V, indexing="ij")
    for i, (el, az, lab) in enumerate([(24, -60, "front"), (18, 20, "side (P-V)"), (62, -125, "top")]):
        x = fig.add_subplot(2, 3, i + 1, projection="3d")
        surf = x.plot_surface(VV, TT, P, cmap="viridis", rstride=3, cstride=1, linewidth=0, antialiased=True, vmin=-3, vmax=26)
        x.set_xlabel("V (nL)", labelpad=6)
        x.set_ylabel("activation time t (ms)", labelpad=8)
        x.set_zlabel("P (mmHg)", labelpad=6)
        x.view_init(elev=el, azim=az)
        x.set_box_aspect(None, zoom=0.9)
        x.set_title(f"view {i + 1}: {lab} (elev {el}, azim {az})", fontsize=10)
    sel = (V >= 0.40) & (V <= 1.05)
    TT2, VV2 = np.meshgrid(t_ms, V[sel], indexing="ij")
    x = fig.add_subplot(2, 3, 4, projection="3d")
    x.plot_surface(VV2, TT2, P[:, sel], cmap="viridis", rstride=3, cstride=1, linewidth=0, vmin=-3, vmax=26)
    x.set_xlabel("V (nL)", labelpad=6)
    x.set_ylabel("activation time t (ms)", labelpad=8)
    x.set_zlabel("P (mmHg)", labelpad=6)
    x.view_init(elev=24, azim=-60)
    x.set_box_aspect(None, zoom=0.9)
    x.set_title("view 1 of the working range 0.40-1.05 nL (z up to 26 mmHg)", fontsize=10)
    x = fig.add_subplot(2, 3, (5, 6))
    mesh = x.pcolormesh(t_ms, V, P.T, cmap="viridis", norm=SymLogNorm(linthresh=1.0, vmin=-3, vmax=P.max()), shading="auto")
    x.contour(t_ms, V, P.T, levels=[0], colors="w", linewidths=0.8)
    if cfg.SOLVER == "PINN":  # entries outside the trained pressure range (clipped), sweep grid
        sweep = np.load(cfg.output / "pvt" / "job0of1.npz")
        clipped = (sweep["clipped_below"] | sweep["clipped_above"])[sweep["done"]].astype(float)
        if clipped.any():
            ts = np.r_[sweep["t_grid_ms"], cfg.BCL_MS]  # later times are passive: the last column repeats
            clipped = np.c_[clipped, clipped[:, -1]]
            x.contourf(ts, sweep["V_grid_nL"][sweep["done"]], clipped, levels=[0.5, 1.5], colors="none", hatches=["///"])
            x.contour(ts, sweep["V_grid_nL"][sweep["done"]], clipped, levels=[0.5], colors="k", linewidths=0.6)
    x.axvline(cfg.T0_MS, color="w", ls=":", lw=1)
    fig.colorbar(mesh, ax=x, label="P (mmHg), symmetric log scale")
    x.set(xlabel="activation time t (ms)", ylabel="V (nL)",
          title=f"P(V, t) map: {len(V)} volumes {V.min():.2f}-{V.max():.2f} nL x {len(t_ms)} times 0-{t_ms.max():.0f} ms "
                "(white: P = 0; dotted: t0" + ("; hatched: outside the trained pressure range)" if cfg.SOLVER == "PINN" else ")"))
    cax = fig.add_axes([0.005, 0.56, 0.008, 0.3])
    fig.colorbar(surf, cax=cax, label="P (mmHg), 3D colour clipped at 26")
    fig.suptitle(f"P-V-t table of the {cfg.SOLVER} atrium read by ngspice (one cycle; total P = active table + passive base)",
                 fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.05, hspace=0.15, wspace=0.08)
    fig.savefig(cfg.output / "pvt_table.png", dpi=120)
    plt.close(fig)
    print("wrote", cfg.output / "pvt_table.png")


def cell_F(X, C, u):
    """Deformation gradient per linear tetrahedron from the nodal displacement."""
    Xc = X[C]
    Dinv = np.linalg.inv(np.stack([Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0], Xc[:, 3] - Xc[:, 0]], 2))
    uc = u[C]
    dU = np.stack([uc[:, 1] - uc[:, 0], uc[:, 2] - uc[:, 0], uc[:, 3] - uc[:, 0]], 2)
    return np.eye(3) + dU @ Dinv


def field_summary(cfg):
    """Summary of the fields of the coupled cycle (OUTPUT_DIR/replay/fields_summary.csv and .txt), with their sanity ranges:
    - sarcomere length ls = lr |F f0| within the working range of cardiac muscle (~1.6-2.3 um; the active law gives
      almost no tension below l0 + 0.003 um);
    - fiber shortening of the order of the cube root of the volume change;
    - J = det F close to 1 (linear-tetrahedron values of the vertex displacements of the P2 field);
    - in-plane wall stress of the order of the Laplace estimate P r / (2 h) of a thin sphere with the same cavity and
      wall volume (cells >= BASE_EXCLUDE_UM from the base)."""
    REPLAY = cfg.output / "replay"
    r = np.load(REPLAY / "pv_results.npz")
    if "region" in r.files:  # WALL = "myo_endo"
        return field_summary_myo_endo(cfg, r)
    X, C, f0, vol, far = r["points"], r["cells"], r["f0"], r["cell_volume"], r["far_from_base"].astype(bool)
    w = vol[far] / vol[far].sum()
    wall_nL = vol.sum() * 1e3  # mm^3 -> nL
    rows = []
    for k in np.argsort(r["t_ms"]):
        F = cell_F(X, C, r["u"][k])
        ls = cfg.LR * np.linalg.norm(np.einsum("cij,cj->ci", F, f0), axis=1)
        J = np.linalg.det(F)
        V, P = float(r["V_nL"][k]), float(r["P_kPa"][k])
        r_mm = (3 * V * 1e-3 / (4 * np.pi)) ** (1 / 3)
        h_mm = ((3 * (V + wall_nL) * 1e-3 / (4 * np.pi)) ** (1 / 3)) - r_mm
        rows.append({"state": int(r["state"][k]), "t_ms": float(r["t_ms"][k]), "V_nL": V, "P_kPa": P,
                     "u_max_um": 1e3 * float(np.linalg.norm(r["u"][k], axis=1).max()),
                     "ls_p5": float(np.percentile(ls, 5)), "ls_mean": float((w * ls[far]).sum()), "ls_p95": float(np.percentile(ls, 95)),
                     "J_min": float(J.min()), "J_max": float(J.max()), "J_mean": float((vol * J).sum() / vol.sum()),
                     "J_far_off_fraction": float(vol[far][np.abs(J[far] - 1) > 0.2].sum() / vol[far].sum()),
                     "sff_wall": float((w * r["sigma_ff"][k][far]).sum()), "sCC_wall": float((w * r["sigma_CC"][k][far]).sum()),
                     "sLL_wall": float((w * r["sigma_LL"][k][far]).sum()), "sRR_wall": float((w * r["sigma_RR"][k][far]).sum()),
                     "laplace_kPa": P * r_mm / (2 * h_mm), "r_um": 1e3 * r_mm, "h_um": 1e3 * h_mm,
                     "T_mean_kPa": float(r["T_mean_kPa"][k])})
    with open(REPLAY / "fields_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in q.items()} for q in rows)
    a = {k: np.array([q[k] for q in rows]) for k in rows[0]}
    Vmax, Vmin = a["V_nL"].max(), a["V_nL"].min()
    i = int(np.argmax(a["P_kPa"]))
    lines = [
        f"cavity V {Vmin:.3f}-{Vmax:.3f} nL, cube-root shortening {100 * (1 - (Vmin / Vmax) ** (1 / 3)):.1f} %; wall-mean ls "
        f"{a['ls_mean'].min():.3f}-{a['ls_mean'].max():.3f} um (shortening {100 * (1 - a['ls_mean'].min() / a['ls_mean'].max()):.1f} %); "
        f"cell ls 5-95 % over the cycle {a['ls_p5'].min():.3f}-{a['ls_p95'].max():.3f} um",
        f"u_max {a['u_max_um'].max():.1f} um (cavity radius {a['r_um'].min():.0f}-{a['r_um'].max():.0f} um, wall ~{a['h_um'].mean():.0f} um)",
        f"J: volume mean {a['J_mean'].min():.4f}-{a['J_mean'].max():.4f}, cells {a['J_min'].min():.3f}-{a['J_max'].max():.3f}; "
        f"away from the base |J - 1| > 0.2 in at most {100 * a['J_far_off_fraction'].max():.2f} % of the wall",
        f"at peak P ({a['P_kPa'][i]:.3f} kPa, t {a['t_ms'][i]:.0f} ms): wall-mean sigma_CC {a['sCC_wall'][i]:.2f}, sigma_ff "
        f"{a['sff_wall'][i]:.2f}, sigma_LL {a['sLL_wall'][i]:.2f}, sigma_RR {a['sRR_wall'][i]:.2f} kPa; in-plane mean "
        f"{(a['sCC_wall'][i] + a['sLL_wall'][i]) / 2:.2f} vs Laplace P r / 2h {a['laplace_kPa'][i]:.2f} kPa",
    ]
    (REPLAY / "fields_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def field_summary_myo_endo(cfg, r):
    """field_summary for the wall of myocardium + endocardium (WALL = "myo_endo"): sarcomere length and wall means of
    the myocardium, the means of the layer on a line of their own, J of all cells, and the Laplace estimate with the
    whole wall (layer + myocardium) against its in-plane mean."""
    REPLAY = cfg.output / "replay"
    X, C, f0, vol, far_all = r["points"], r["cells"], r["f0"], r["cell_volume"], r["far_from_base"].astype(bool)
    myo = r["region"] == 1
    far, lay = far_all & myo, far_all & ~myo
    w, wl, wa = vol[far] / vol[far].sum(), vol[lay] / vol[lay].sum(), vol[far_all] / vol[far_all].sum()
    wall_nL = vol.sum() * 1e3  # mm^3 -> nL
    rows = []
    for k in np.argsort(r["t_ms"]):
        F = cell_F(X, C, r["u"][k])
        ls = cfg.LR * np.linalg.norm(np.einsum("cij,cj->ci", F[myo], f0[myo]), axis=1)
        J = np.linalg.det(F)
        V, P = float(r["V_nL"][k]), float(r["P_kPa"][k])
        r_mm = (3 * V * 1e-3 / (4 * np.pi)) ** (1 / 3)
        h_mm = ((3 * (V + wall_nL) * 1e-3 / (4 * np.pi)) ** (1 / 3)) - r_mm
        rows.append({"state": int(r["state"][k]), "t_ms": float(r["t_ms"][k]), "V_nL": V, "P_kPa": P,
                     "u_max_um": 1e3 * float(np.linalg.norm(r["u"][k], axis=1).max()),
                     "ls_p5": float(np.percentile(ls, 5)), "ls_mean": float((w * ls[far[myo]]).sum()), "ls_p95": float(np.percentile(ls, 95)),
                     "J_min": float(J.min()), "J_max": float(J.max()), "J_mean": float((vol * J).sum() / vol.sum()),
                     "J_far_off_fraction": float(vol[far_all][np.abs(J[far_all] - 1) > 0.2].sum() / vol[far_all].sum()),
                     "sff_wall": float((w * r["sigma_ff"][k][far]).sum()), "sCC_wall": float((w * r["sigma_CC"][k][far]).sum()),
                     "sLL_wall": float((w * r["sigma_LL"][k][far]).sum()), "sRR_wall": float((w * r["sigma_RR"][k][far]).sum()),
                     "sCC_endo": float((wl * r["sigma_CC"][k][lay]).sum()), "sLL_endo": float((wl * r["sigma_LL"][k][lay]).sum()),
                     "sRR_endo": float((wl * r["sigma_RR"][k][lay]).sum()),
                     "inplane_whole_wall": float((wa * 0.5 * (r["sigma_CC"][k] + r["sigma_LL"][k])[far_all]).sum()),
                     "laplace_kPa": P * r_mm / (2 * h_mm), "r_um": 1e3 * r_mm, "h_um": 1e3 * h_mm,
                     "T_mean_kPa": float(r["T_mean_kPa"][k])})
    with open(REPLAY / "fields_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in q.items()} for q in rows)
    a = {k: np.array([q[k] for q in rows]) for k in rows[0]}
    Vmax, Vmin = a["V_nL"].max(), a["V_nL"].min()
    i = int(np.argmax(a["P_kPa"]))
    lines = [
        f"cavity (lumen) V {Vmin:.3f}-{Vmax:.3f} nL, cube-root shortening {100 * (1 - (Vmin / Vmax) ** (1 / 3)):.1f} %; "
        f"myocardial wall-mean ls {a['ls_mean'].min():.3f}-{a['ls_mean'].max():.3f} um (shortening "
        f"{100 * (1 - a['ls_mean'].min() / a['ls_mean'].max()):.1f} %); myocardial cell ls 5-95 % over the cycle "
        f"{a['ls_p5'].min():.3f}-{a['ls_p95'].max():.3f} um",
        f"u_max {a['u_max_um'].max():.1f} um (cavity radius {a['r_um'].min():.0f}-{a['r_um'].max():.0f} um, wall incl. layer "
        f"~{a['h_um'].mean():.0f} um)",
        f"J (all cells): volume mean {a['J_mean'].min():.4f}-{a['J_mean'].max():.4f}, cells {a['J_min'].min():.3f}-"
        f"{a['J_max'].max():.3f}; away from the base |J - 1| > 0.2 in at most {100 * a['J_far_off_fraction'].max():.2f} % of the wall",
        f"at peak P ({a['P_kPa'][i]:.3f} kPa, t {a['t_ms'][i]:.0f} ms): myocardial wall-mean sigma_CC {a['sCC_wall'][i]:.2f}, "
        f"sigma_ff {a['sff_wall'][i]:.2f}, sigma_LL {a['sLL_wall'][i]:.2f}, sigma_RR {a['sRR_wall'][i]:.2f} kPa; in-plane mean "
        f"of the whole wall {a['inplane_whole_wall'][i]:.2f} vs Laplace P r / 2h {a['laplace_kPa'][i]:.2f} kPa",
        f"endocardial layer at peak P: mean sigma_CC {a['sCC_endo'][i]:.2f}, sigma_LL {a['sLL_endo'][i]:.2f}, sigma_RR "
        f"{a['sRR_endo'][i]:.2f} kPa; most compressed over the cycle: sigma_CC {a['sCC_endo'].min():.2f} kPa "
        f"(t {a['t_ms'][int(np.argmin(a['sCC_endo']))]:.0f} ms)",
    ]
    (REPLAY / "fields_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cycle_movie(cfg, plt):
    """OUTPUT_DIR/pv_cycle.mp4: the coupled cycle state by state (the replay states, from the atrial activation onset):
    left the deforming wall on a long-axis section through the apex (mesh lines, coloured by the fiber stress), right
    the atrial P-V loop of the coupled cycle with the current state. Same layout as the movie of the volume_constrain
    examples. Needs vtk and ffmpeg (params.FFMPEG); skipped if ffmpeg is not found."""
    from matplotlib.collections import PolyCollection
    writer = ffmpeg_writer(cfg, plt)
    if writer is None:
        return
    RUN = cfg.output / "circuit"
    r = np.load(cfg.output / "replay" / "pv_results.npz")
    c = windkessel.last_cycle(dict(np.load(RUN / "result.npz")), json.loads((RUN / "params.json").read_text())["cycle"])
    X, cells = r["points"] / cfg.MM_PER_UNIT, r["cells"]
    section, o, e1, n = long_axis_section(cfg, X, cells)

    sff = r["sigma_ff"]
    lim = float(np.ceil(np.percentile(np.abs(sff[:, r["far_from_base"].astype(bool)]), 99.5)))  # whole kPa
    t, V, P, T = r["t_ms"], r["V_nL"], r["P_kPa"] * MMHG, r["T_mean_kPa"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    pc = PolyCollection([], cmap="RdBu_r", edgecolors="#00000026", linewidths=0.15)
    pc.set_clim(-lim, lim)
    a1.add_collection(pc)
    ext = np.concatenate([((X + 1e3 * r["u"][i]) - o) @ np.c_[e1, n] for i in range(len(t))])
    a1.set_xlim(ext[:, 0].min() - 5, ext[:, 0].max() + 5)
    a1.set_ylim(ext[:, 1].max() + 5, -5)  # base on top, apex at the bottom
    a1.set_aspect("equal")
    a1.set(xlabel="µm", ylabel="distance from the base plane (µm)")
    fig.colorbar(pc, ax=a1, shrink=0.8, extend="both", label="fiber stress σ$_{ff}$ (kPa)")
    a2.plot(c["V_la"], c["P_la"], "-", color=BLUE, lw=2)
    a2.scatter(V, P, s=12, color=BLUE)
    a2.axhline(0, color=GREY, lw=0.6)
    dot, = a2.plot([], [], "o", ms=12, color=ORANGE, mec="white", mew=1.5)
    a2.set(xlabel="atrial volume (nL)", ylabel="atrial pressure (mmHg)")
    a2.grid(alpha=0.3)
    label = a2.set_title("atrial P-V loop of the coupled cycle\n ", loc="left", fontsize=10)  # two lines reserved
    title = a1.set_title("", loc="left", fontsize=10)
    fig.tight_layout()
    with writer.saving(fig, str(cfg.output / "pv_cycle.mp4"), dpi=130):
        for i in np.argsort(t):
            polys, vals = section(X + 1e3 * r["u"][i], sff[i])
            pc.set_verts(polys)
            pc.set_array(vals)
            dot.set_data([V[i]], [P[i]])
            title.set_text(f"{cfg.SOLVER}, t = {t[i]:.0f} ms: long-axis section through the apex")
            label.set_text(f"atrial P-V loop of the coupled cycle\nV = {V[i]:.3f} nL,  P = {P[i]:.2f} mmHg,  T = {T[i]:.1f} kPa")
            writer.grab_frame()
    plt.close(fig)
    print("wrote", cfg.output / "pv_cycle.mp4", flush=True)


def windkessel_figures(cfg, group):
    """Steps figures_table (after table), figures_cycle (after circuit), figures_fields (after replay)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    RUN = cfg.output / "circuit"
    if group == "table":
        table_figure(cfg, plt)
    elif group == "cycle":
        p = json.loads((RUN / "params.json").read_text())
        m = json.loads((RUN / "metrics.json").read_text())
        c = windkessel.last_cycle(dict(np.load(RUN / "result.npz")), p["cycle"])
        rows = metric_rows(cfg, m)
        with open(cfg.output / "metrics.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["part", "metric", "value", "measured (3 dpf)"])
            w.writerows(rows)
        for r in rows:
            print(" | ".join(r))
        pv_figure(cfg, plt, c, m)
        flow_figure(cfg, plt, c, m, p)
        pressure_volume_figure(cfg, plt, c, m)
    elif group == "fields":
        field_summary(cfg)
        cycle_movie(cfg, plt)
    else:
        raise ValueError(f"unknown figure group {group}: table, cycle or fields")


# ==================== invert_para ====================
def invert_comparison(cfg, trk, us, result, ev):
    """OUTPUT_DIR/comparison.png: long-axis section through the apex at ED and PS, tracking vs model, with C and T_MAX."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    lm = json.loads((cfg.input / "base_plane.json").read_text())
    o, n, apex = (np.asarray(lm[k], float) for k in ("origin", "normal", "apex"))
    n /= np.linalg.norm(n)
    n *= np.sign((apex - o) @ n)
    e1 = (apex - o) - ((apex - o) @ n) * n
    e1 /= np.linalg.norm(e1)
    normal = np.cross(n, e1)

    def cut(x, tri):
        d = (x - o) @ normal
        segs = []
        for t in tri[(d[tri].max(1) > 0) & (d[tri].min(1) < 0)]:
            s = d[t]
            segs.append([x[t[i]] + s[i] / (s[i] - s[j]) * (x[t[j]] - x[t[i]])
                         for i, j in ((0, 1), (1, 2), (2, 0)) if (s[i] > 0) != (s[j] > 0)])
        p = np.array(segs)
        return np.stack(((p - o) @ e1, (p - o) @ n), -1)
    fam = result["fibers_angle_deg_share"]

    def deg(a):  # (endo, epi) angle of a family (one number if the same at every depth; older files: one number)
        a = [a] if np.ndim(a) == 0 else list(a)
        return f"{a[0]:g}" if len(a) == 1 or a[0] == a[1] else f"{a[0]:g} to {a[1]:g} (endo to epi)"
    fib = ("two fibers " + " / ".join(deg(a) for a, _ in fam) + " deg") if len(fam) > 1 else f"one fiber {deg(fam[0][0])} deg"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, (name, k) in zip(axes, (("ED", cfg.ED_FRAME), ("PS", cfg.PS_FRAME))):
        for side in ("endo", "epi"):
            ax.add_collection(LineCollection(cut(trk.track[k], trk.tri[side]), colors="k", linewidths=2.2,
                                             label="tracking" if side == "endo" else None))
            ax.add_collection(LineCollection(cut(trk.X + us[k], trk.sim_tri[side]), colors="#eb6834", linewidths=1.6,
                                             label="model" if side == "endo" else None))
        e = ev["frames"][name]
        ax.autoscale()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"t{k} ({name})", loc="left", fontsize=12)
        ax.set_title(f"distance endo {e['endo_rms_um']:.1f} / epi {e['epi_rms_um']:.1f} um   "
                     f"DSC wall {e['wall_DSC']:.2f} / cavity {e['cavity_DSC']:.2f}", loc="right", fontsize=9, color="0.35")
        ax.set_xlabel("um")
        ax.legend(frameon=False, loc="lower left", fontsize=10)
    axes[0].set_ylabel("distance from the base plane (um)")
    fig.suptitle(f"C = {result['C_kPa']:.4f} kPa     T_max = {result['T_MAX_kPa']:.3f} kPa     ({fib})", fontsize=14)
    fig.savefig(cfg.output / "comparison.png", dpi=150, facecolor="white")
    plt.close(fig)
