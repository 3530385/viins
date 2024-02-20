# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyins
from utils.data_transforms import transfom_reference, transform_imu

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 12

# %%
reference_trajectory = pd.read_csv(
    "pyins/examples/data/reference_trajectory.csv", index_col="time"
)
imu = pd.read_csv("pyins/examples/data/imu.csv", index_col="time")
gnss = pd.read_csv("pyins/examples/data/gnss.csv", index_col="time")

# %%
reference_trajectory


# %%
leica_reference = transfom_reference(
    pd.read_csv("mav0/state_groundtruth_estimate0/data.csv")
)
leica_reference = leica_reference[leica_reference > 35]
leica_reference
# %%
imu = transform_imu(pd.read_csv("mav0/imu0/data.csv"))

imu[["accel_x", "accel_y", "accel_z"]].iloc[:-1000].plot()
# %%
imu_to_ant_b = np.array([0.645, 0.000, 0.000])
position_meas = pyins.measurements.Position(
    leica_reference[leica_reference.index > 0.2], 1.0, imu_to_ant_b
)
position_meas.data
# velocity_meas = pyins.measurements.NedVelocity(leica_reference, 0.3, imu_to_ant_b)
# import yaml

# with open("mav0/imu0/sensor.yaml") as file:
#     sensor_config = yaml.safe_load(file)
# print(sensor_config["T_BS"]["data"])
#
# position_meas.data[
#     [
#         "lat",
#     ]
# ].plot()

# %%
increments = pyins.strapdown.compute_increments_from_imu(imu, "rate")

# %%
position_sd = 5.0
velocity_sd = 0.5
level_sd = 0.2
azimuth_sd = 1.0

# %%
pva_error = pyins.sim.generate_pva_error(
    position_sd, velocity_sd, level_sd, azimuth_sd, 0
)

pva_initial = pyins.sim.perturb_pva(reference_trajectory.iloc[0], pva_error)

# %%


gyro_model = pyins.inertial_sensor.EstimationModel(
    bias_sd=300.0 * pyins.transform.DH_TO_RS,
    noise=1.0 * pyins.transform.DRH_TO_RRS,
    bias_walk=30.0 * pyins.transform.DH_TO_RS / 60,
)

accel_model = pyins.inertial_sensor.EstimationModel(
    bias_sd=0.05, noise=0.1 / 60, bias_walk=0.01 / 60
)

# %%
result = pyins.filters.run_feedback_filter(
    pva_initial,
    position_sd,
    velocity_sd,
    level_sd,
    azimuth_sd,
    increments,
    gyro_model,
    accel_model,
    measurements=[
        position_meas,
    ],
)

# %%
plt.plot(result.innovations["Position"], label=["lat", "lon", "alt"])
plt.xlabel("System time, s")
plt.title("Position normalized innovations")
plt.legend()

# %%
trajectory_error = pyins.transform.compute_state_difference(
    result.trajectory, reference_trajectory
)

# %%
plt.figure(figsize=(10, 10))
for i, col in enumerate(trajectory_error.columns, start=1):
    plt.subplot(3, 3, i)
    plt.plot(trajectory_error[col], label="error")
    plt.plot(3 * result.trajectory_sd[col], "k", label="3-sigma bounds")
    plt.plot(-3 * result.trajectory_sd[col], "k")
    plt.legend()
    plt.title(col)

plt.suptitle("Trajectory errors with 3-sigma bounds")
plt.tight_layout()

# %%
plt.plot(result.gyro * pyins.transform.RS_TO_DH, label=["bias_x", "bias_y", "bias_z"])
plt.legend()
plt.title("Gyro bias estimates, deg/hour")
plt.xlabel("System time, s")

# %%
plt.plot(result.accel, label=["bias_x", "bias_y", "bias_z"])
plt.legend()
plt.title("Accel bias estimates, m/s^2")
plt.xlabel("System time, s")

# %%
trajectory_error[["north", "east", "down"]].plot()

# %%
result.trajectory[["lat"]].plot()

# %%
leica_reference.lat.plot()
leica_reference
