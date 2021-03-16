from django.conf import settings
import os

def processing(dataset_name):
    if not os.path.isdir(settings.BASE_DIR + "/api/models/" + dataset_name):
        return ""
    current_model = ""

    if os.path.isfile(settings.BASE_DIR + "/api/logs/model_log.txt"):
        with open(settings.BASE_DIR + "/api/logs/model_log.txt", "r") as f:
            lines = f.readlines()
        for l in lines:
            if l.split('/')[0] == dataset_name:
                current_model = l.split('/')[1][:-1]

    return {"list": os.listdir(settings.BASE_DIR + "/api/models/" + dataset_name), "current": current_model}