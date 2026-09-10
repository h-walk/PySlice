import numpy as np

from pyslice.md import analyze_md_trajectory


def test_md_analysis_splits_temperature_and_energy_plots_for_runaway_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLBACKEND", "Agg")

    time_ps = np.linspace(0.0, 10.0, 500)
    temperature = 300.0 + 18.0 * time_ps + 4.0 * np.sin(time_ps)
    potential_energy = -100.0 + 0.5 * time_ps
    kinetic_energy = 5.0 + 0.15 * time_ps
    total_energy = potential_energy + kinetic_energy
    data = np.column_stack(
        (
            np.arange(len(time_ps)),
            time_ps,
            temperature,
            potential_energy,
            kinetic_energy,
            total_energy,
        )
    )

    log_file = tmp_path / "production.log"
    np.savetxt(
        log_file,
        data,
        header=(
            "ORB MD Production\n"
            "Temperature: 300 K\n"
            "Step Time(ps) Temp(K) Epot(eV) Ekin(eV) Etot(eV)"
        ),
    )

    temperature_plot = tmp_path / "md_analysis.png"
    analyze_md_trajectory(
        trajectory_file=tmp_path / "not-needed.traj",
        log_file=log_file,
        output_file=temperature_plot,
    )

    energy_plot = tmp_path / "md_analysis_energy.png"
    assert temperature_plot.exists()
    assert temperature_plot.stat().st_size > 0
    assert energy_plot.exists()
    assert energy_plot.stat().st_size > 0
