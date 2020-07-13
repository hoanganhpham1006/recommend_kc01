from django.conf import settings
import os

def processing(dataset_name):
    if not os.path.isdir(settings.BASE_DIR + "/api/models/" + dataset_name):
        return ""
    with open(settings.BASE_DIR + "/api/logs/model_log.txt", "r") as f:
        lines = f.readlines()
    i = 1
    current_model = ""
    while i < len(lines):
        if lines[-i].split('/')[0] == dataset_name:
            current_model = lines[-i].split('/')[1][:-1]
        i += 1
    return {"list": os.listdir(settings.BASE_DIR + "/api/models/" + dataset_name), "current": current_model}