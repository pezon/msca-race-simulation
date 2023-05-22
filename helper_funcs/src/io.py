import pickle
import shutil
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
try:
    from google.colab import files
except ImportError:
    files = None

PREPROCESSOR_FILE = "preprocessor_reinforcement_{race}.pkl"
POLICY_TFLITE_FILE = "nn_reinforcement_{race}.tflite"


def download_archive(archive_path):
    if files is not None:
        files.download(archive_path)
        # try again if this fails:
        # https://github.com/googlecolab/colabtools/issues/469


def export_archive(dirname, base_filename):
  return shutil.make_archive(base_filename, "zip", dirname)


def import_archive(dirname):
  if files is None:
    return
  uploaded = files.upload()
  for fn in uploaded.keys():
    print(f"User uploaded file {fn} with length {len(uploaded[fn])} bytes")
    shutil.rmtree(dirname)
    zip_files = ZipFile(BytesIO(uploaded[fn]), "r")
    zip_files.extractall(dirname)
    zip_files.close()


def save_preprocessor(env, save_dir, race):
    save_dir.mkdir(exist_ok=True, parents=True)
    saved_preprocessor = save_dir / PREPROCESSOR_FILE.format(race=race)
    with open(saved_preprocessor, "wb") as f:
        pickle.dump(env.cat_preprocessor, f)
    # print(f"Saved preprocessor: {saved_preprocessor}")
    return saved_preprocessor


def save_policy_tflite(policy_dir, save_dir, race):
    """Save policy as TFLite.

    Converts Q Network to TFlite.
    See [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert) for more details.
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(policy_dir))
    tflite_policy = converter.convert()
    exported_qnet_tflite = save_dir / POLICY_TFLITE_FILE.format(race=race)
    with open(exported_qnet_tflite, "wb") as f:
        f.write(tflite_policy)
    return exported_qnet_tflite


def evaluate_policy(
    tf_env,
    py_env,
    policy,
    num_episodes = 3,
    print_lap_decisions: bool = True
):
    total_return = 0.0
    for i in range(num_episodes):
        episode_return = 0.0
        time_step = tf_env.reset()
        lap = 1
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_return += np.mean(time_step.reward)
            if print_lap_decisions:
                print(f"race {i + 1}: driver = {py_env.idx_driver}, lap = {lap}, action = {action_step.action[0]}")
            lap += 1
        total_return += episode_return
        final_position = py_env.race.positions[
           py_env.race.get_last_compl_lap(py_env.idx_driver),
           py_env.idx_driver
        ]
        if print_lap_decisions:
            print(f"race {i + 1}: driver = {py_env.idx_driver}, final_position = {final_position}")
    average_return = total_return / num_episodes
    return average_return
 