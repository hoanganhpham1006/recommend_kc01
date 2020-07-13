from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from api.forms.dter_training_form import DTERTrainForm
from api.helpers.dter_training_helper import processing
from api.helpers.response_format import json_format
import json

class DTERTrainView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'Starting DTER Train Success'
    failure = 'Starting DTER Train Failed'

    def post(self, request):
        mydata = json.loads(request.body.decode("utf-8"))
        # list_params = ['start_date', 'end_date']
        # for param in list_params:
        #     if param not in mydata:
        #         return JsonResponse("JSON Params must have " + param, status=422, safe=False)
        if 'start_date' in mydata:
            start_date = mydata['start_date']
        else:
            start_date = None
        
        if 'end_date' in mydata:
            end_date = mydata['end_date']
        else:
            end_date = None
        
        if 'force_train' in mydata:
            if mydata['force_train'] == "True" or mydata['force_train'] == "False":
                force_train = bool(mydata['force_train'])
            else:
                JsonResponse("JSON Params force_train must be True or False", status=422, safe=False)
        else:
            force_train = False
        return self._format_response(start_date, end_date, force_train)
    
    def _format_response(self, start_date, end_date, force_train):
        result = processing(start_date, end_date, force_train)
        return json_format(code=200, message=self.success, data=result, errors=None)
  