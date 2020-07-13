from django import forms
from api.forms.abstract_form import AbstractForm

class DTERTrainForm(AbstractForm):
    start_date = forms.CharField(required=False)
    end_date = forms.CharField(required=False)
    force_train = forms.CharField(required=False)