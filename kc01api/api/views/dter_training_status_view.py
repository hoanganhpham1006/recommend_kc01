from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from api.forms.dter_training_status_form import DTERTrainStatusForm
from api.helpers.dter_training_status_helper import processing
from api.helpers.response_format import json_format
import json

class DTERTrainStatusView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'Check Training Status Success'
    failure = 'Check Training Status Failed'

    def post(self, request):
        mydata = json.loads(request.body.decode("utf-8"))
        # list_params = ['start_date']
        # for param in list_params:
        #     if param not in mydata:
        #         return JsonResponse("JSON Params must have " + param, status=422)
        return self._format_response()
    
    def _format_response(self):
        result = processing()
        return json_format(code=200, message=self.success, data=result, errors=None)
  