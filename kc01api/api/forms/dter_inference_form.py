from django import forms
from api.forms.abstract_form import AbstractForm

class DTERInferForm(AbstractForm):
    list_url = forms.CharField(required=True)