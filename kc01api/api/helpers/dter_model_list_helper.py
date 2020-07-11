from django.conf import settings
import os

def processing(dataset_name):
    if not os.path.isdir(settings.BASE_DIR + "/api/models/" + dataset_name):
        return ""
    return os.listdir(settings.BASE_DIR + "/api/models/" + dataset_name)