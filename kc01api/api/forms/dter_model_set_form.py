from django import forms
from api.forms.abstract_form import AbstractForm

class DTERModelSetForm(AbstractForm):
    model_name = forms.CharField(required=True)
    dataset_name = forms.CharField(required=True)