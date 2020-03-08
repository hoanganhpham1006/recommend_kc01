from django.urls import path
from api.views.example_view import ExampleView
from api.views.dter_inference_view import DTERInferView
app_name = 'api'

urlpatterns = [
    path('example', ExampleView.as_view(), name='example'),
    path('dter_inference', DTERInferView.as_view(), name='dter_inference'),
]

