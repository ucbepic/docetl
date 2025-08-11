import modal

APP_NAME = "docetl-experiments"
VOLUME_NAME = "docetl-ro-experiments"
VOLUME_MOUNT_PATH = "/mnt/docetl-ro-experiments"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True) 

