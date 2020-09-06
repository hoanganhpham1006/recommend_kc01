from django import forms
from api.forms.abstract_form import AbstractForm

class DTERTrainForm(AbstractForm):
    dataset_name = forms.CharField(required=True)
    start_date = forms.CharField(required=False)
    end_date = forms.CharField(required=False)
    force_train = forms.CharField(required=False)