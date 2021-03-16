from django import forms
from api.forms.abstract_form import AbstractForm

class DTERInferForm(AbstractForm):
    dataset_name = forms.CharField(required=True)
    list_url = forms.CharField(required=True)