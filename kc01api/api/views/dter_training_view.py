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
        list_params = ['start_date', 'end_date']
        for param in list_params:
            if param not in mydata:
                return JsonResponse("JSON Params must have " + param, status=422, safe=False)
        start_date = mydata['start_date']
        end_date = mydata['end_date']
        return self._format_response(start_date, end_date)
    
    def _format_response(self, start_date, end_date):
        result = processing(start_date, end_date)
        return json_format(code=200, message=self.success, data=result, errors=None)
  