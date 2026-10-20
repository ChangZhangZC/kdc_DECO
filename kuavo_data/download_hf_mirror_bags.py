import os
import re
import time
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from huggingface_hub import HfApi

# =========================================================
# Environment variables
# =========================================================
# 使用中国大陆常见的 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 可选：把 huggingface 的少量元数据缓存放到临时目录
# os.environ["HF_HOME"] = "/bayes-tmp/huggingface"
# os.environ["HF_HUB_CACHE"] = "/bayes-tmp/huggingface/hub"

# =========================================================
# Config
# =========================================================
HF_ENDPOINT = os.environ["HF_ENDPOINT"].rstrip("/")

REPO_ID = "LejuRobotics/kuavo_data_challenge_icra"
REPO_TYPE = "dataset"
REVISION = "main"

# 你截图中的目录
# TARGET_DIR = "sim/TASK1-ToySorting"
# TARGET_DIR = "sim/TASK2-ParcelWeighing"
TARGET_DIR = "sim/TASK3-ConveyorBeltSorting"

# 只下载前 300 个
# 2026.4.10 代码改动为从后向前下载600个
NUM_DOWNLOAD = 600 

# 保存到这个目录，且平铺成 *.bag
# LOCAL_DIR = "/root/bayes-tmp/kuavo_dataset/icra_task1"
# LOCAL_DIR = "/root/bayes-tmp/kuavo_dataset/icra_task2"
LOCAL_DIR = "/root/bayes-tmp/kuavo_dataset/icra_task3"

# 默认多线程
NUM_WORKERS = 8

# 单次读取块大小
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

# 超时设置
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120
MAX_RETRIES = 10

# =========================================================
# Helpers
# =========================================================
def natural_sort_key(path: str):
    name = os.path.basename(path)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def build_download_url(repo_id: str, repo_type: str, revision: str, file_path: str) -> str:
    # 镜像下载地址格式与 Hugging Face 一致
    if repo_type == "dataset":
        return f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/{revision}/{file_path}"
    elif repo_type == "model":
        return f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/{file_path}"
    elif repo_type == "space":
        return f"{HF_ENDPOINT}/spaces/{repo_id}/resolve/{revision}/{file_path}"
    else:
        raise ValueError(f"Unsupported repo_type: {repo_type}")


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=NUM_WORKERS, pool_maxsize=NUM_WORKERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =========================================================
# Prepare target dir
# =========================================================
os.makedirs(LOCAL_DIR, exist_ok=True)

# =========================================================
# List files from mirror endpoint
# =========================================================
print("Fetching file list from mirror...")

api = HfApi(endpoint=HF_ENDPOINT)

repo_info = api.repo_info(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    revision=REVISION,
    files_metadata=True,
)

bag_files = []
for f in repo_info.siblings:
    repo_path = getattr(f, "rfilename", None)
    file_size = getattr(f, "size", 0)

    if repo_path and repo_path.startswith(TARGET_DIR + "/") and repo_path.endswith(".bag"):
        bag_files.append((repo_path, int(file_size or 0)))

bag_files = sorted(bag_files, key=lambda x: natural_sort_key(x[0]))

if NUM_DOWNLOAD is not None:
    bag_files = bag_files[-NUM_DOWNLOAD:]

total_files = len(bag_files)
total_bytes = sum(size for _, size in bag_files)

print(f"Found {total_files} bag files")
print(f"Total size: {sizeof_fmt(total_bytes)}")
print(f"Saving to: {LOCAL_DIR}")

# =========================================================
# Shared progress bar
# =========================================================
pbar = tqdm(
    total=total_bytes,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
    desc="Downloading",
    dynamic_ncols=True,
)

# =========================================================
# Download one file
# =========================================================
def download_one(entry):
    repo_path, expected_size = entry
    filename = os.path.basename(repo_path)

    final_path = os.path.join(LOCAL_DIR, filename)
    part_path = final_path + ".part"

    # 已存在且大小匹配则跳过
    if os.path.exists(final_path) and expected_size > 0 and os.path.getsize(final_path) == expected_size:
        pbar.update(expected_size)
        return f"skip  {filename}"

    # 清理旧的 part 文件
    if os.path.exists(part_path):
        os.remove(part_path)

    url = build_download_url(REPO_ID, REPO_TYPE, REVISION, repo_path)
    session = make_session()

    downloaded_this_attempt = 0
    try:
        with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
            resp.raise_for_status()

            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded_this_attempt += len(chunk)
                    pbar.update(len(chunk))

        actual_size = os.path.getsize(part_path)
        if expected_size > 0 and actual_size != expected_size:
            # 回滚进度条
            if downloaded_this_attempt > 0:
                pbar.update(-downloaded_this_attempt)
            os.remove(part_path)
            raise RuntimeError(
                f"size mismatch for {filename}: got {actual_size}, expected {expected_size}"
            )

        os.replace(part_path, final_path)
        return f"done  {filename}"

    except Exception:
        # 回滚这次已经累加到总进度条的字节
        if downloaded_this_attempt > 0:
            pbar.update(-downloaded_this_attempt)

        if os.path.exists(part_path):
            os.remove(part_path)
        raise
    finally:
        session.close()


# =========================================================
# Run multi-thread download
# =========================================================
errors = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(download_one, entry): entry for entry in bag_files}

    for future in as_completed(futures):
        entry = futures[future]
        try:
            future.result()
        except Exception as e:
            errors.append((entry[0], str(e)))

pbar.close()

# =========================================================
# Summary
# =========================================================
if errors:
    print("\nSome files failed:")
    for path, err in errors[:20]:
        print(f"- {path}: {err}")
    print(f"\nFailed: {len(errors)} / {total_files}")
    raise SystemExit(1)

print("\nAll downloads completed successfully.")