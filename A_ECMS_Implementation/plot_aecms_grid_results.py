import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path setup for loading model components from repository root.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from A_ECMS_Implementation.aecms_controller import AECMS_Controller


def _fmt3_no_round(value):
    """Format to x.xxx using truncation (no rounding up)."""
    return f"{np.trunc(float(value) * 1000.0) / 1000.0:.3f}"


def _has_final_soc_column(csv_path):
    """Check quickly whether CSV has a final_soc column."""
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns
        return "final_soc" in cols
    except Exception:
        return False


def resolve_input_csv(root_dir):
    """Resolve best input CSV path for plotting.

    Priority:
    1) CLI argument path, if provided.
    2) Existing known result files, preferring those with final_soc,
       then by newest modification time.
    """
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if not os.path.isabs(arg_path):
            arg_path = os.path.join(root_dir, arg_path)
        if not os.path.exists(arg_path):
            print(f"Error: Input CSV not found: {arg_path}")
            sys.exit(1)
        return arg_path

    candidates = [
        os.path.join(root_dir, "aecms_grid_results.csv"),
        os.path.join(root_dir, "A_ECMS_Implementation", "aecms_grid_results.csv"),
        os.path.join(root_dir, "aecms_grid_results2.csv"),
        os.path.join(root_dir, "aecms_grid_results1.csv"),
    ]
    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        tried = "\n".join(candidates)
        print(f"Error: No A-ECMS result CSV found. Tried:\n{tried}")
        sys.exit(1)

    with_soc = [p for p in existing if _has_final_soc_column(p)]
    if with_soc:
        return max(with_soc, key=os.path.getmtime)
    return max(existing, key=os.path.getmtime)


def calculate_willans_reference(root_dir):
    """Compute Willans marginal-cost line using AECMS controller calibration method."""
    vmap_path = os.path.join(root_dir, "Engine", "325kW.vmap")
    vem_path = os.path.join(root_dir, "Emotor", "P2_Group5_EM.vem")
    vemo_path = os.path.join(root_dir, "Emotor", "EM_Map - kopie.vemo")
    vreess_path = os.path.join(root_dir, "Emotor", "P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(root_dir, "Emotor", "REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(root_dir, "Emotor", "REESS_Internal_Resistance.vbatr")

    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)

    ctrl = AECMS_Controller(truck)

    p_fit_watts = np.asarray(ctrl.willans_p_fit_watts, dtype=float)
    f_fit_gs = np.asarray(ctrl.willans_f_fit_gs, dtype=float)

    if p_fit_watts.size == 0:
        raise ValueError("No valid points found for Willans-line fit.")

    p_line_watts = np.linspace(0.0, float(ctrl.willans_p_max_watts), 200)
    f_line_gs = ctrl.willans_k * p_line_watts + ctrl.willans_intercept
    f_line_gs = np.maximum(f_line_gs, 0.0)

    return {
        "p_fit_kw": p_fit_watts / 1000.0,
        "f_fit_gs": f_fit_gs,
        "p_line_kw": p_line_watts / 1000.0,
        "f_line_gs": f_line_gs,
        "p_max_kw": float(ctrl.willans_p_max_watts) / 1000.0,
        "k": ctrl.willans_k,
        "intercept": ctrl.willans_intercept,
        "s_dis_willans": ctrl.s_dis_willans,
        "s_chg_willans": ctrl.s_chg_willans,
    }


def load_grid_data(csv_path):
    """Load grid-search results and reshape into 2D arrays for plotting."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"kp_dis", "kp_chg", "fuel", "dev"}
    missing = required_cols.difference(df.columns)
    if missing:
        print(f"Error: Missing required columns: {sorted(missing)}")
        sys.exit(1)

    pivot_fuel = df.pivot(index="kp_dis", columns="kp_chg", values="fuel")
    pivot_dev = df.pivot(index="kp_dis", columns="kp_chg", values="dev")

    kp_dis_ax = pivot_fuel.index.values.astype(float)
    kp_chg_ax = pivot_fuel.columns.values.astype(float)

    X, Y = np.meshgrid(kp_chg_ax, kp_dis_ax)
    fuel = pivot_fuel.values
    dev = pivot_dev.values

    final_soc = None
    if "final_soc" in df.columns:
        pivot_soc = df.pivot(index="kp_dis", columns="kp_chg", values="final_soc")
        final_soc = pivot_soc.values

    return kp_dis_ax, kp_chg_ax, X, Y, fuel, dev, final_soc


def _argmin_fuel_with_mask(fuel, mask):
    masked_fuel = np.where(mask, fuel, np.inf)
    return np.unravel_index(np.argmin(masked_fuel), fuel.shape)


def find_optimal_point(X, Y, fuel, dev, final_soc=None, target_soc=0.30, target_dev=1.0):
    """Find optimum prioritizing minimum fuel while enforcing end-SOC constraint."""
    from scipy.interpolate import RegularGridInterpolator

    fuel_interp = RegularGridInterpolator(
        (Y[:, 0], X[0, :]),
        fuel,
        bounds_error=False,
        fill_value=None,
    )

    if final_soc is not None:
        above_mask = final_soc >= target_soc

        if np.any(above_mask):
            # Fuel-first among solutions that satisfy final SOC >= target SOC.
            idx_opt = _argmin_fuel_with_mask(fuel, above_mask)
        else:
            max_soc = np.max(final_soc)
            closest_below_mask = np.isclose(final_soc, max_soc, atol=1e-12)
            idx_opt = _argmin_fuel_with_mask(fuel, closest_below_mask)
            print(
                f"Warning: No points end at or above SOC {target_soc:.2f}. "
                "Using closest-below SOC point."
            )
    else:
        min_dev = np.min(dev)
        close_dev_mask = np.isclose(dev, min_dev, atol=1e-12)
        idx_opt = _argmin_fuel_with_mask(fuel, close_dev_mask)
        print(
            "Note: CSV has no final_soc column, so above/below 30% cannot be enforced exactly. "
            "Using minimum deviation (closest to 30%) then minimum fuel."
        )

    opt_kp_chg = X[idx_opt]
    opt_kp_dis = Y[idx_opt]
    min_fuel = float(fuel_interp((opt_kp_dis, opt_kp_chg)))

    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    if final_soc is not None:
        contour = ax_temp.contour(X, Y, final_soc, levels=[target_soc], alpha=0)
    else:
        contour = ax_temp.contour(X, Y, dev, levels=[target_dev], alpha=0)

    if not contour.allsegs or not contour.allsegs[0]:
        x_line = np.array([])
        y_line = np.array([])
        fuel_line = np.array([])
    else:
        verts = contour.allsegs[0][0]
        x_line = verts[:, 0]
        y_line = verts[:, 1]
        pts = np.column_stack((y_line, x_line))
        fuel_line = fuel_interp(pts)
    plt.close(fig_temp)

    opt_soc = None
    if final_soc is not None:
        opt_soc = float(final_soc[idx_opt])

    return opt_kp_dis, opt_kp_chg, min_fuel, opt_soc, x_line, y_line, fuel_line


def plot_grid_results(
    kp_dis_ax,
    kp_chg_ax,
    X,
    Y,
    fuel,
    dev,
    final_soc,
    opt_point,
    boundary_line,
    output_path,
    target_soc=0.30,
    target_dev=0.8,
):
    """Plot 3D optimization surface and 2D map, matching calibration style."""
    opt_kp_dis, opt_kp_chg, opt_fuel, opt_soc = opt_point
    line_x, line_y, line_fuel = boundary_line

    try:
        from scipy.interpolate import RegularGridInterpolator

        kp_chg_fine = np.linspace(
            float(np.nanmin(kp_chg_ax)),
            float(np.nanmax(kp_chg_ax)),
            max(220, len(kp_chg_ax) * 8),
        )
        kp_dis_fine = np.linspace(
            float(np.nanmin(kp_dis_ax)),
            float(np.nanmax(kp_dis_ax)),
            max(220, len(kp_dis_ax) * 8),
        )
        X3d, Y3d = np.meshgrid(kp_chg_fine, kp_dis_fine)

        fuel_interp = RegularGridInterpolator(
            (kp_dis_ax, kp_chg_ax),
            fuel,
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_pts = np.column_stack((Y3d.ravel(), X3d.ravel()))
        fuel3d = fuel_interp(interp_pts).reshape(X3d.shape)
    except Exception:
        X3d, Y3d, fuel3d = X, Y, fuel

    fig = plt.figure(figsize=(18, 6.5))

    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(
        X3d,
        Y3d,
        fuel3d,
        cmap="viridis",
        alpha=1.0,
        rcount=fuel3d.shape[0],
        ccount=fuel3d.shape[1],
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=True,
        zorder=1,
    )
    surf.set_rasterized(True)
    surf.set_edgecolor("none")
    surf.set_linewidth(0.0)

    mesh_stride_r = max(1, fuel3d.shape[0] // 26)
    mesh_stride_c = max(1, fuel3d.shape[1] // 26)
    ax1.plot_wireframe(
        X3d,
        Y3d,
        fuel3d,
        rstride=mesh_stride_r,
        cstride=mesh_stride_c,
        color="white",
        linewidth=0.20,
        alpha=0.22,
    )

    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    cbar1.set_label("Spotřeba paliva [kg]", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    ax1.plot(
        [opt_kp_chg],
        [opt_kp_dis],
        [opt_fuel],
        color="red",
        marker="*",
        markersize=16,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Optimální bod",
        zorder=100,
        clip_on=False,
    )

    if len(line_x) > 0:
        line_label = (
            f"Omezení SOC={int(target_soc * 100)}%"
            if final_soc is not None
            else f"Omezení odchylky={target_dev:.1f}%"
        )
        ax1.plot(
            line_x,
            line_y,
            line_fuel + 0.06,
            color="black",
            linewidth=4,
            label=line_label,
            zorder=10,
        )

    # Camera and aspect tuned for readability over wide Kp ranges.
    ax1.view_init(elev=28, azim=42)
    ax1.set_box_aspect((1.25, 1.15, 0.55))
    ax1.set_xlabel("Zesílení nabíjení Kp_chg", fontsize=14)
    ax1.set_ylabel("Zesílení vybíjení Kp_dis", fontsize=14)
    ax1.set_zlabel("Spotřeba paliva [kg]", fontsize=14)
    ax1.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax1.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    z_min = float(np.nanmin(fuel))
    z_max = float(np.nanmax(fuel))
    z_pad = max(0.02, 0.08 * (z_max - z_min))
    ax1.set_zlim(z_min - z_pad, z_max + z_pad)
    ax1.set_title("A-ECMS 3D optimalizační plocha", fontsize=17)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.legend(loc="upper right", fontsize=12)

    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(X, Y, fuel, levels=20, cmap="viridis")
    cbar2 = plt.colorbar(cf, ax=ax2)
    cbar2.set_label("Spotřeba paliva [kg]", fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    if len(line_x) > 1:
        line_label = (
            f"Omezení SOC={int(target_soc * 100)}%"
            if final_soc is not None
            else f"Omezení odchylky={target_dev:.1f}%"
        )
        ax2.plot(
            line_x,
            line_y,
            "k-",
            linewidth=3,
            label=line_label,
        )

    if final_soc is not None:
        for soc_iso in [max(0.0, target_soc - 0.05), min(1.0, target_soc + 0.05)]:
            ax2.contour(X, Y, final_soc, levels=[soc_iso], colors="k", linestyles=":", linewidths=1.5)
    else:
        for dev_iso in [0.5, 1.2]:
            ax2.contour(X, Y, dev, levels=[dev_iso], colors="k", linestyles=":", linewidths=1.5)

    ax2.plot(
        opt_kp_chg,
        opt_kp_dis,
        "r*",
        markersize=17,
        markeredgecolor="white",
        label="Optimální bod",
    )

    ax2.set_xlabel("Zesílení nabíjení Kp_chg", fontsize=14)
    ax2.set_ylabel("Zesílení vybíjení Kp_dis", fontsize=14)
    ax2.set_title("A-ECMS optimalizační mapa", fontsize=17)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.legend(loc="upper right", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_willans_standalone(willans_ref, output_path):
    """Vykresli samostatný graf Willansovy přímky v češtině."""
    fig = plt.figure(figsize=(6.4, 4.6))
    ax = fig.add_subplot(111)

    ax.scatter(
        willans_ref["p_fit_kw"],
        willans_ref["f_fit_gs"],
        s=12,
        c="tab:blue",
        alpha=0.35,
        edgecolors="none",
        label="Vzorky mapy motoru",
    )
    ax.plot(
        willans_ref["p_line_kw"],
        willans_ref["f_line_gs"],
        color="tab:red",
        linewidth=2.3,
        label="Lineární fit Willansovy přímky",
    )

    ax.set_xlabel("Mechanicky vykon motoru [kW]", fontsize=12)
    ax.set_ylabel("Prutok paliva [g/s]", fontsize=12)
    ax.set_title("Willansova přímka (mezní náklad)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xlim(0.0, willans_ref["p_max_kw"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    info = (
        f"Sklon k = {willans_ref['k']:.3e} [g/J]\n"
        f"s_dis (Willans) = {willans_ref['s_dis_willans']:.3f}\n"
        f"s_chg (Willans) = {willans_ref['s_chg_willans']:.3f}"
    )
    ax.text(
        0.97,
        0.03,
        info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90},
    )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = resolve_input_csv(root_dir)
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(root_dir, f"{csv_stem}_plots.pdf")
    willans_out_path = os.path.join(root_dir, f"{csv_stem}_willans_line.pdf")
    target_soc = 0.30
    target_dev = 0.8

    print("Computing Willans line reference from A-ECMS controller...")
    willans_ref = calculate_willans_reference(root_dir)

    print(f"Loading A-ECMS grid data from: {csv_path}")
    kp_dis, kp_chg, X, Y, fuel, dev, final_soc = load_grid_data(csv_path)

    print("Finding constrained optimum...")
    opt_kp_dis, opt_kp_chg, min_fuel, opt_soc, lx, ly, lf = find_optimal_point(
        X,
        Y,
        fuel,
        dev,
        final_soc=final_soc,
        target_soc=target_soc,
        target_dev=target_dev,
    )
    if opt_soc is not None:
        print(
            "Optimal factors: "
            f"Kp_dis={opt_kp_dis:.3f}, Kp_chg={opt_kp_chg:.3f} -> "
            f"Fuel={_fmt3_no_round(min_fuel)} kg, Final SOC={opt_soc * 100.0:.2f}%"
        )
    else:
        print(
            "Optimal factors: "
            f"Kp_dis={opt_kp_dis:.3f}, Kp_chg={opt_kp_chg:.3f} -> Fuel={_fmt3_no_round(min_fuel)} kg"
        )

    print("Plotting...")
    plot_grid_results(
        kp_dis,
        kp_chg,
        X,
        Y,
        fuel,
        dev,
        final_soc,
        (opt_kp_dis, opt_kp_chg, min_fuel, opt_soc),
        (lx, ly, lf),
        output_path=out_path,
        target_soc=target_soc,
        target_dev=target_dev,
    )

    print("Vykresluji samostatný graf Willansovy přímky...")
    plot_willans_standalone(willans_ref, willans_out_path)


if __name__ == "__main__":
    main()
