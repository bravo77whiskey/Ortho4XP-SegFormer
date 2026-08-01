import queue
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Tile_Utils as TILE
from O4_Cfg_Vars import cfg_app_vars, gui_app_vars_short


def _quiet_ui(monkeypatch):
    monkeypatch.setattr(TILE.UI, "red_flag", False)
    monkeypatch.setattr(TILE.UI, "vprint", lambda *args, **kwargs: None)
    monkeypatch.setattr(TILE.UI, "progress_bar", lambda *args, **kwargs: None)


def _completed_producer():
    event = threading.Event()
    event.set()
    return event


def test_concurrent_download_setting_is_visible_and_bounded():
    setting = cfg_app_vars["max_download_slots"]

    assert "max_download_slots" in gui_app_vars_short
    assert setting["short_name"] == "Concurrent texture downloads"
    assert setting["values"] == tuple(range(1, 9))
    assert setting["default"] == 1


def test_download_queue_finishes_a_delayed_retry(monkeypatch):
    _quiet_ui(monkeypatch)

    class DelayedRetryQueue(queue.Queue):
        delay_retries = False

        def put(self, item, *args, **kwargs):
            if self.delay_retries and isinstance(item, tuple):
                time.sleep(0.05)
            return super().put(item, *args, **kwargs)

    attempts = 0

    def fail_once(_tile, _task_id):
        nonlocal attempts
        attempts += 1
        return attempts >= 2

    monkeypatch.setattr(TILE.IMG, "build_jpeg_ortho", fail_once)
    download_queue = DelayedRetryQueue()
    convert_queue = queue.Queue()
    tile = object()
    download_queue.put((7,))
    download_queue.delay_retries = True

    result = TILE.download_textures(
        tile,
        download_queue,
        convert_queue,
        workers=2,
        producer_done_event=_completed_producer(),
    )

    assert result == 1
    assert attempts == 2
    assert convert_queue.get_nowait() == (tile, 7)
    assert download_queue.unfinished_tasks == 0
    assert download_queue.empty()


def test_download_queue_waits_for_late_producer_work(monkeypatch):
    _quiet_ui(monkeypatch)
    processed = []
    monkeypatch.setattr(
        TILE.IMG,
        "build_jpeg_ortho",
        lambda _tile, task_id: processed.append(task_id) or 1,
    )
    download_queue = queue.Queue()
    convert_queue = queue.Queue()
    producer_done = threading.Event()
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            TILE.download_textures(
                object(),
                download_queue,
                convert_queue,
                workers=2,
                producer_done_event=producer_done,
            )
        )
    )
    thread.start()
    download_queue.put((11,))
    producer_done.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [1]
    assert processed == [11]
    assert convert_queue.qsize() == 1


def test_permanent_failure_is_retried_three_times_and_reported(monkeypatch):
    _quiet_ui(monkeypatch)
    attempts = 0

    def always_fail(_tile, _task_id):
        nonlocal attempts
        attempts += 1
        return 0

    monkeypatch.setattr(TILE.IMG, "build_jpeg_ortho", always_fail)
    download_queue = queue.Queue()
    convert_queue = queue.Queue()
    download_queue.put((13,))

    result = TILE.download_textures(
        object(),
        download_queue,
        convert_queue,
        workers=2,
        producer_done_event=_completed_producer(),
    )

    assert result == 0
    assert attempts == 3
    assert convert_queue.empty()
    assert download_queue.unfinished_tasks == 0


def test_cancellation_drains_the_queue_without_downloading(monkeypatch):
    _quiet_ui(monkeypatch)
    monkeypatch.setattr(TILE.UI, "red_flag", True)
    calls = []
    monkeypatch.setattr(
        TILE.IMG,
        "build_jpeg_ortho",
        lambda *_args: calls.append(True) or 1,
    )
    download_queue = queue.Queue()
    convert_queue = queue.Queue()
    download_queue.put((1,))
    download_queue.put((2,))

    result = TILE.download_textures(
        object(),
        download_queue,
        convert_queue,
        workers=2,
        producer_done_event=_completed_producer(),
    )

    assert result == 0
    assert calls == []
    assert download_queue.unfinished_tasks == 0
