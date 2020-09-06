from django import forms
from api.forms.abstract_form import AbstractForm

class DTERTrainStatusForm(AbstractForm):
    dataset_name = forms.CharField(required=True)
    query_id = forms.CharField(required=False)