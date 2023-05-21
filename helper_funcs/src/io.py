import pickle
import shutil
from io import BytesIO
from zipfile import ZipFile

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
    save_dir.mkdir(exist_ok=True, parents=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(policy_dir), signature_keys=["action"])
    tflite_policy = converter.convert()
    exported_qnet_tflite = save_dir / POLICY_TFLITE_FILE.format(race=race)
    with open(exported_qnet_tflite, "wb") as f:
        f.write(tflite_policy)
    return exported_qnet_tflite