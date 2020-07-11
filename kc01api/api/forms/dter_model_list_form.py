from django import forms
from api.forms.abstract_form import AbstractForm

class DTERModelListForm(AbstractForm):
    dataset_name = forms.CharField(required=True)