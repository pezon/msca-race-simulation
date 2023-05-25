from functools import partial
from time import sleep

import fastf1
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import pandas as pd

N_LAPS = 55
FPS = 4.0
DURATION = N_LAPS / FPS

TEAM_MAP = {
    "ToroRosso": "alphatauri",
}

TIRE_MAP = {
    "A6": "SOFT",
    "A4": "MEDIUM",
    "A3": "HARD",
}


def mcs_analysis(
        race_results: list,
        use_print_result: bool,
        use_plot: bool
):

    # ------------------------------------------------------------------------------------------------------------------
    # PREPROCESSING ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not type(race_results) is list and type(race_results[0]) is dict:
        raise RuntimeError("List of dicts required as result_objs (list of results from race.race_results())!")

    no_sim_runs = len(race_results)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE DATAFRAME CONTAINING THE PROCESSED RESULT POSITIONS -------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create pandas dataframe in the form [driver_initials, no_pos1, no_pos2, ...] for cumulated results
    driver_initials = list(race_results[0]["driverinfo"].keys())
    no_drivers = len(driver_initials)
    col_names = ["no_pos" + str(i) for i in range(1, no_drivers + 1)]

    race_results_df = pd.DataFrame(np.zeros((no_drivers, no_drivers), dtype=np.int32),
                                   columns=col_names,
                                   index=driver_initials)

    # count number of positions for current driver
    for initials in driver_initials:
        tmp_pos_results = [0] * no_drivers

        for idx_race in range(no_sim_runs):
            cur_result_pos = int(race_results[idx_race]["driverinfo"][initials]["positions"][-1])
            tmp_pos_results[cur_result_pos - 1] += 1

        # add tmp_results to driver's row in pandas dataframe
        race_results_df.loc[initials] = tmp_pos_results

    # ------------------------------------------------------------------------------------------------------------------
    # PRINT MEAN POSITIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_print_result:
        print("RESULT: Mean positions after %i simulation runs..." % no_sim_runs)
        mean_posis = []

        for idx_driver in range(no_drivers):
            no_pos = race_results_df.iloc[idx_driver].values
            mean_pos = np.sum(no_pos * np.arange(1, no_drivers + 1)) / no_sim_runs

            mean_posis.append([driver_initials[idx_driver], mean_pos])

        # sort list by mean position
        mean_posis.sort(key=lambda x: x[1])

        # print list
        for entry in mean_posis:
            print("RESULT: %s: %.1f" % (entry[0], entry[1]))

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if use_plot:
        subplots_per_row = 3
        no_rows = int(np.ceil(no_drivers / subplots_per_row))

        fig, axes = plt.subplots(no_rows, subplots_per_row, sharex="all", sharey="all")

        # create one subplot per driver
        cur_col = 0
        cur_row = 0

        for idx_driver in range(no_drivers):
            # use bar plots to show position distributions
            axes[cur_row][cur_col].\
                bar(range(1, no_drivers + 1),
                    list(race_results_df.iloc[idx_driver] / no_sim_runs * 100.0),
                    tick_label=range(1, no_drivers + 1))

            # add driver initials above plot
            axes[cur_row][cur_col].set_title(driver_initials[idx_driver])

            # set ylabel for first plot per row
            if cur_col == 0:
                axes[cur_row][cur_col].set_ylabel("percentage")

            # set xlabel for last plot per column
            if cur_row == no_rows - 1:
                axes[cur_row][cur_col].set_xlabel("rank position")

            # count up column and row indices
            cur_col += 1
            if cur_col >= subplots_per_row:
                cur_col = 0
                cur_row += 1

        fig.suptitle("Distribution of final positions (%i simulated races)" % no_sim_runs)
        plt.show()


def plot_position_changes_from_sim(
    sim,
    at_lap = 1,
    ax = None,
    drivers = None,
    n_laps = None,
):
    for driver in drivers:
        positions = sim["driverinfo"][driver]["positions"]
        team = sim["driverinfo"][driver]["team"]
        team = TEAM_MAP.get(team, team)
        color = fastf1.plotting.team_color(team)
        ax.plot(
            np.arange(1, at_lap + 1),
            positions[0:at_lap],
            label=driver,
            color=color)
    ax.set_xlim([1, n_laps])
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_title("Position")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    # legend 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_tire_strategy_from_sim(sim, at_lap = 1, ax = None, drivers = None, n_laps = None):
    # create stints
    stints = []
    for driver in drivers:
        for stint_idx, stint in enumerate(sim["driverinfo"][driver]["strategy_info"]):
            try:
                out_lap = sim["driverinfo"][driver]["strategy_info"][stint_idx + 1][0]
            except IndexError:
                out_lap = n_laps + 1
            if stint[0] > at_lap:
                continue
            if out_lap > at_lap:
                out_lap = at_lap
            stints.append({
                "Driver": driver,
                "Stint": stint_idx + 1,
                "Compound": TIRE_MAP.get(stint[1]),
                "StintLength": out_lap - stint[0],
                "InLap": stint[0],
                "OutLap": out_lap
            })
    stints = pd.DataFrame(stints)
    # plot stints for each driver
    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars
            ax.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                edgecolor="black",
                fill=True
            )
            previous_stint_end += row["StintLength"]
    # styling
    ax.set_title("Tire")
    ax.set_xlabel("Lap")
    ax.set_xlim([0, n_laps])
    ax.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


def make_frame(t, sim, fps, n_laps, ax_tire, ax_pos, fig, drivers):
    lap = int(t * fps)
    # clear plot
    ax_tire.clear()
    ax_pos.clear()
    # plot again
    plot_tire_strategy_from_sim(sim, at_lap=lap, ax=ax_tire, drivers=drivers, n_laps=n_laps)
    plot_position_changes_from_sim(sim, at_lap=lap, ax=ax_pos, drivers=drivers, n_laps=n_laps)
    fig.suptitle("Shanghai Grand Prix 2019 Simulation")
    plt.tight_layout()
    # be kind to the api
    sleep(0.05)
    return mplfig_to_npimage(fig)


def create_simulation_video(sim, drivers, n_laps = None, fps = FPS, duration = DURATION):
        # plot
        fig, (ax_tire, ax_pos) = plt.subplots(1, 2, figsize=(16.0, 6.0))

        # creating animation
        animation = VideoClip(
            partial(
                make_frame,
                fps=fps,
                sim=sim,
                n_laps=n_laps,
                ax_tire=ax_tire,
                ax_pos=ax_pos,
                fig=fig,
                drivers=drivers),
            duration=duration)
        return animation
        # displaying animation with auto play and looping
        # animation.ipython_display(fps=FPS, loop=True, autoplay=True)


def plot_position_changes_from_api(laps, at_lap = None, ax = None, drivers = None):
    if at_lap is None:
        at_lap = laps["LapNumber"].max()
    for drv in drivers:
        drv_laps = laps.pick_driver(drv)
        driver = drv_laps["Driver"].iloc[0]
        team = drv_laps["Team"].iloc[0]
        color = fastf1.plotting.team_color(team)
        ax.plot(
            drv_laps["LapNumber"][0:at_lap],
            drv_laps["Position"][0:at_lap],
            label=driver,
            color=color)
    ax.set_xlim([1, 55])
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_title("Position")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    # legend 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_tire_strategy_from_api(laps, at_lap = None, ax = None, drivers = None):
    if at_lap is None:
        at_lap = laps["LapNumber"].max()
    laps = laps[laps["LapNumber"] <= at_lap]
    # create stints
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    # plot stints for each driver
    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars
            ax.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                edgecolor="black",
                fill=True
            )
            previous_stint_end += row["StintLength"]
    # styling
    ax.set_title("Tire")
    ax.set_xlabel("Lap")
    ax.set_xlim([0, 55])
    ax.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax



# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
