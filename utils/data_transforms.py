import pandas as pd
import numpy as np
from pyins.transform import perturb_lla

START_TIMESTAMP = 1403636580838555648
START_LAT = 47.37667
START_LON = 8.54770
START_ALT = 150.499


def quaternion_to_rph(qw, qx, qy, qz):
    """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

    Returns:
        yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
        pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, pi/2]`
        roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]`

    The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

    Note:
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
    """
    if 2 * (qx * qz - qw * qy) >= 0.94:  # Preventing gimbal lock for north pole
        yaw = np.arctan2(qx * qy - qw * qz, qx * qz + qw * qy)
        roll = 0
    elif 2 * (qx * qz - qw * qy) <= -0.94:  # Preventing gimbal lock for south pole
        yaw = -np.arctan2(qx * qy - qw * qz, qx * qz + qw * qy)
        roll = 0
    else:
        yaw = np.arctan2(qy * qz + qw * qx, 1 / 2 - (qx**2 + qy**2))
        roll = np.arctan2(qx * qy - qw * qz, 1 / 2 - (qy**2 + qz**2))
    pitch = np.arcsin(-2 * (qx * qz - qw * qy))

    return 180 + yaw, pitch, roll


def transfom_reference(df: pd.DataFrame):
    time_name = "#timestamp"
    drop_columns = [
        " b_w_RS_S_x [rad s^-1]",
        " b_w_RS_S_y [rad s^-1]",
        " b_w_RS_S_z [rad s^-1]",
        " b_a_RS_S_y [m s^-2]",
        " b_a_RS_S_z [m s^-2]",
        " b_a_RS_S_x [m s^-2]",
        "qw",
        "qx",
        "qy",
        "qz",
        "x",
        "y",
        "z",
    ]
    df = df.rename(
        columns={
            time_name: "time",
            " v_RS_R_x [m s^-1]": "VN",
            " v_RS_R_y [m s^-1]": "VE",
            " v_RS_R_z [m s^-1]": "VD",
            " p_RS_R_x [m]": "x",
            " p_RS_R_y [m]": "y",
            " p_RS_R_z [m]": "z",
            " q_RS_w []": "qw",
            " q_RS_x []": "qx",
            " q_RS_y []": "qy",
            " q_RS_z []": "qz",
        }
    )
    df["z"] *= -1
    df["y"] *= -1
    df["time"] = df["time"] / 1e9
    df["time"] = df["time"] - START_TIMESTAMP / 1e9
    df[["heading", "pitch", "roll"]] = df[["qw", "qx", "qy", "qz"]].apply(
        lambda x: quaternion_to_rph(**x), axis=1, result_type="expand"
    )
    df[["lat", "lon", "alt"]] = df[["x", "y", "z"]].apply(
        lambda x: perturb_lla(
            np.array([START_LAT, START_LON, START_ALT]), x.to_numpy()
        ),
        axis=1,
        result_type="expand",
    )

    return df.drop(drop_columns, axis=1).set_index("time")


def transform_imu(df: pd.DataFrame):
    time_name = "#timestamp [ns]"
    df = df.rename(
        columns={
            time_name: "time",
            "w_RS_S_x [rad s^-1]": "gyro_z",
            "w_RS_S_y [rad s^-1]": "gyro_y",
            "w_RS_S_z [rad s^-1]": "gyro_x",
             "a_RS_S_x [m s^-2]": "accel_z",
            "a_RS_S_y [m s^-2]": "accel_y",
            "a_RS_S_z [m s^-2]": "accel_x",
        }
    )
    #df["gyro_x"] *= -1
    #df["accel_x"] *= -1
    df["gyro_z"] *= -1
    df["accel_z"] *= -1
    df["time"] = df["time"] / 1e9
    df["time"] = df["time"] - START_TIMESTAMP / 1e9
    return df[df["time"] > 0].set_index("time")
