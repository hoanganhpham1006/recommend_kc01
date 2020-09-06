from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from api.forms.dter_inference_form import DTERInferForm
from api.helpers.dter_inference_helper import processing
from api.helpers.response_format import json_format
import json
from api import supported_dataset, status

class DTERInferView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'DTER Infer Success'
    failure = 'DTER Infer Failed'

    def post(self, request):
        mydata = json.loads(request.body.decode("utf-8"))
        if 'list_url' not in mydata or 'dataset_name' not in mydata:
            return JsonResponse("JSON Params must have list_url and dataset_name", status=422)
        list_url = mydata['list_url']
        dataset_name = mydata['dataset_name']
        if dataset_name not in supported_dataset:
            return JsonResponse("Supported datasets: " + str(supported_dataset), status=422)
        if dataset_name not in status.keys():
            return JsonResponse("Trained model is not existed: " + str(supported_dataset), status=422)
        return self._format_response(list_url, dataset_name)
    
    def _format_response(self, list_url, dataset_name):
        result = processing(list_url, dataset_name)
        return json_format(code=200, message=self.success, data=result, errors=None)
  