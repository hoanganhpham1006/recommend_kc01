from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from api.forms.dter_training_status_form import DTERTrainStatusForm
from api.helpers.dter_training_status_helper import processing
from api.helpers.response_format import json_format
import json
from api import supported_dataset

class DTERTrainStatusView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'Check Training Status Success'
    failure = 'Check Training Status Failed'

    def post(self, request):
        mydata = json.loads(request.body.decode("utf-8"))
        if 'dataset_name' not in mydata:
            return JsonResponse("JSON Params must have dataset_name", status=422)
        dataset_name = mydata['dataset_name']
        if dataset_name not in supported_dataset:
            return JsonResponse("Supported datasets: " + supported_dataset, status=422)
        # list_params = ['start_date']
        # for param in list_params:
        #     if param not in mydata:
        #         return JsonResponse("JSON Params must have " + param, status=422)
        return self._format_response(dataset_name)
    
    def _format_response(self, dataset_name):
        status, result = processing(dataset_name)
        if status == 0:
            return json_format(code=201, message=self.success, data=result, errors=None)
        else:
            return json_format(code=200, message=self.success, data=result, errors=None)
  