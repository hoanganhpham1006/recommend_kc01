from django import forms
from api.forms.abstract_form import AbstractForm

class DTERTrainStatusForm(AbstractForm):
    query_id = forms.CharField(required=False)