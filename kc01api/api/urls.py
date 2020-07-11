from django.urls import path
from api.views.example_view import ExampleView
from api.views.dter_inference_view import DTERInferView
from api.views.dter_training_view import DTERTrainView
from api.views.dter_training_status_view import DTERTrainStatusView
from api.views.dter_model_list_view import DTERModelListView
from api.views.dter_model_set_view import DTERModelSetView
app_name = 'api'

urlpatterns = [
    path('example', ExampleView.as_view(), name='example'),
    path('dter_inference', DTERInferView.as_view(), name='dter_inference'),
    path('dter_training', DTERTrainView.as_view(), name='dter_training'),
    path('dter_training_status', DTERTrainStatusView.as_view(), name='dter_training_status'),
    path('dter_model_list', DTERModelListView.as_view(), name='dter_model_list'),
    path('dter_model_set', DTERModelSetView.as_view(), name='dter_model_set'),
]