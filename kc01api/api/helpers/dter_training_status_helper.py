from api.helpers.log_status_helper import status_from_logd
from django.conf import settings
import os

def processing(dataset_name):
    if dataset_name == "MostPortal":
        dataset_name = "Most_Portal"
        train_log_file = MOST_TRAIN_LOG
    elif dataset_name == "QNPortal":
        dataset_name = "QN_Portal"
        train_log_file = QN_TRAIN_LOG
    log_file = settings.BASE_DIR + train_log_file
    if not os.path.isfile(log_file):
        return 1, 100
    status, progress = status_from_logd(log_file)
    return status, progress[:-1]