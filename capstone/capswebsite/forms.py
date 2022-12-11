from dataclasses import field
from faulthandler import disable
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.forms import ModelForm
from django.forms.fields import DateField
from django.contrib.admin.widgets import AdminDateWidget
from .models import *



class DateInput(forms.DateInput):
    input_type = 'date'

#USER ACCOUNTS
class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('email','numberID','password1','password2','firstName','lastName')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["email"].widget.attrs.update(
            {'required': True, 'name': 'email', 'id': 'email', 'type': 'text', 'class': 'form-control', 'placeholder': 'Email'})
        self.fields["firstName"].widget.attrs.update(
            {'required': True, 'name': 'firstName', 'id': 'firstName', 'type': 'text', 'class': 'form-control', 'placeholder': 'First Name'})
        self.fields["lastName"].widget.attrs.update(
            {'required': True, 'name': 'lastName', 'id': 'firstName', 'type': 'text', 'class': 'form-control', 'placeholder': 'Last Name'})
        self.fields["numberID"].widget.attrs.update(
            {'required': True, 'name': 'numberID', 'id': 'numberID', 'class': 'form-control', 'placeholder': 'Student Number'})
        self.fields["password1"].widget.attrs.update(
            {'required': True, 'name': 'password1', 'id': 'password1', 'type': 'password', 'class': 'form-control', 'placeholder': 'Password'})
        self.fields["password2"].widget.attrs.update(
            {'required': True, 'name': 'password2', 'id': 'password2', 'type': 'password', 'class': 'form-control', 'placeholder': 'Confirm Password'})

        if 'college' in self.data:
            try:
                college_id = int(self.data.get('college'))
                self.fields['course'].queryset = Course.objects.filter(college=college_id).order_by('college')
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty City queryset
        elif self.instance.pk:
            self.fields['course'].queryset = self.instance.college.course_set.order_by('college')


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answers
        fields = ('email','numberID','firstName','lastName','college','course','year','block','question1','question2','question3','question4','question5')
        widgets = {
        'student': forms.HiddenInput(),
     }
    question1 = forms.CharField(widget=forms.Textarea,required=False)
    question2 = forms.CharField(widget=forms.Textarea,required=False)
    question3 = forms.CharField(widget=forms.Textarea,required=False)
    question4 = forms.CharField(widget=forms.Textarea,required=False)
    question5 = forms.CharField(widget=forms.Textarea,required=False)
    block = forms.IntegerField(required=False)
    numberID = forms.IntegerField(required=False)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["email"].widget.attrs.update(
            {'required': True, 'name': 'email', 'id': 'email', 'type': 'text', 'class': 'form-control', 'placeholder': 'Email'})
        self.fields["firstName"].widget.attrs.update(
            {'required': False, 'name': 'firstName', 'id': 'firstName', 'type': 'text', 'class': 'form-control', 'placeholder': 'First Name'})
        self.fields["lastName"].widget.attrs.update(
            {'required': False, 'name': 'lastName', 'id': 'firstName', 'type': 'text', 'class': 'form-control', 'placeholder': 'Last Name'})
        self.fields["numberID"].widget.attrs.update(
            {'required': True, 'name': 'numberID', 'id': 'numberID', 'class': 'form-control', 'placeholder': 'Student Number'})
        self.fields["college"].widget.attrs.update(
            {'required': True, 'name': 'college', 'id': 'college', 'class': 'form-control1', 'placeholder': 'College'})
        self.fields["course"].widget.attrs.update(
            {'required': True, 'name': 'course', 'id': 'course', 'class': 'form-control1', 'placeholder': 'Course'})
        self.fields["year"].widget.attrs.update(
            {'required': True, 'name': 'year', 'id': 'year', 'class': 'form-control1', 'placeholder': 'Year'})
        self.fields["block"].widget.attrs.update(
            {'required': True, 'name': 'block', 'id': 'block', 'class': 'form-control1blk', 'placeholder': 'Block'})
        self.fields["question1"].widget.attrs.update(
            {'required': True, 'name': 'question1', 'id': 'question1', 'type': 'text', 'class': 'textarea', 'placeholder': 'Share your experience', 'rows':'10', 'cols':'100', 'onkeyup':'countWords(this);'})
        self.fields["question2"].widget.attrs.update(
            {'required': True, 'name': 'question2', 'id': 'question2', 'type': 'text', 'class': 'textarea', 'placeholder': 'Share your experience', 'rows':'10', 'cols':'100', 'onkeyup':'countWords2(this);'})
        self.fields["question3"].widget.attrs.update(
            {'required': True, 'name': 'questio3', 'id': 'question3', 'type': 'text', 'class': 'textarea', 'placeholder': 'Share your experience', 'rows':'10', 'cols':'100', 'onkeyup':'countWords3(this);'})
        self.fields["question4"].widget.attrs.update(
            {'required': True, 'name': 'question4', 'id': 'question4', 'type': 'text', 'class': 'textarea', 'placeholder': 'Share your experience', 'rows':'10', 'cols':'100', 'onkeyup':'countWords4(this);'})
        self.fields["question5"].widget.attrs.update(
            {'required': True, 'name': 'question5', 'id': 'question5', 'type': 'text', 'class': 'textarea', 'placeholder': 'Share your experience', 'rows':'10', 'cols':'100', 'onkeyup':'countWords5(this);'})
        
        if 'college' in self.data:
            try:
                college_id = int(self.data.get('college'))
                self.fields['course'].queryset = Course.objects.filter(college=college_id).order_by('college')
            except (ValueError, TypeError):
                pass  # invalid input from the client; ignore and fallback to empty City queryset
        elif self.instance.pk:
            self.fields['course'].queryset = self.instance.college.course_set.order_by('college')
