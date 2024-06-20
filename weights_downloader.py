import os
import subprocess
import tarfile
import time

BUCKET_URL = "https://open-source-models.s3.fr-par.scw.cloud"


class WeightsDownloader:
    def __init__(self):
        pass

    def download_weights(self, weights_str, dest="models"):
        self.download_if_not_exists(weights_str, dest)

    def download_if_not_exists(self, weights_str, dest):
        if not os.path.exists(f"{dest}/{weights_str}"):
            self.download(weights_str, dest)

    def download(self, weights_str, dest):
        model_url = f"{BUCKET_URL}/{weights_str}.tar"

        start = time.time()
        print(f"Downloading {weights_str}")
        tar_file_path = f"{dest}/{weights_str}.tar"
        subprocess.check_call(
            ["pget", "-vf", model_url, tar_file_path], close_fds=False
        )
        tar_file_path = f"{dest}/{weights_str}.tar"
        print(f"Extracting {tar_file_path}")
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=dest)
        os.remove(tar_file_path)
        elapsed_time = time.time() - start

        if os.path.isfile(os.path.join(dest, os.path.basename(weights_str))):
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weights_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"Downloaded {weights_str} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        else:
            print(f"Downloaded {weights_str} in {elapsed_time:.2f}s")


if __name__ == "__main__":
    WeightsDownloader().download_if_not_exists("stable-audio-open-1.0", "/tmp/models")
