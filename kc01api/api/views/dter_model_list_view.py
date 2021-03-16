from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from api.forms.dter_model_list_form import DTERModelListForm
from api.helpers.dter_model_list_helper import processing
from api.helpers.response_format import json_format
import json
from api import supported_dataset

class DTERModelListView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'Get Model List Success'
    failure = 'Get Model List Failed'

    def post(self, request):
        mydata = json.loads(request.body.decode("utf-8"))
        list_params = ['dataset_name']
        for param in list_params:
            if param not in mydata:
                return JsonResponse("JSON Params must have " + param, status=422, safe=False)
        dataset_name = mydata['dataset_name']
        if dataset_name not in supported_dataset:
            return JsonResponse("Supported datasets: " + supported_dataset, status=422)
        return self._format_response(dataset_name)
    
    def _format_response(self, dataset_name):
        result = processing(dataset_name)
        if result != "":
            return json_format(code=200, message=self.success, data=result, errors=None)
        else:
            return json_format(code=404, message=self.failure, data="", errors="Does not support " + dataset_name)
  