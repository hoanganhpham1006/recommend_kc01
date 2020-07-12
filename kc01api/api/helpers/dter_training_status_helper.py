from api.helpers.log_status_helper import status_from_logd
from django.conf import settings
import os

def processing():
    log_file = settings.BASE_DIR + "/api/logs/train_log.txt"
    if not os.path.isfile(log_file):
        return "There is not any traning process"
    status, progress = status_from_logd(log_file)
    return progress