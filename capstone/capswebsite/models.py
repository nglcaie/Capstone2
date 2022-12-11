from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.core.validators import RegexValidator
from django.db.models.deletion import CASCADE, DO_NOTHING
from datetime import datetime
from django.utils import timezone
import os, random
from django.utils.html import mark_safe
 
# Create your models here.
class UserManager(BaseUserManager):
    def create_user(self,email, password=None):
        if not email:
            raise ValueError("Users must have an email address")
 
        user = self.model(
            email=self.normalize_email(email),
        )
        user.set_password(password)
        user.save(using=self._db)
        return user
 
    def create_superuser(self,email,password):
        """
        Creates and saves a superuser with the giv email, name and password.
        """
        user = self.create_user(
            email=self.normalize_email(email),
            password=password,
        )
        user.is_active = True
        user.is_admin = True
        user.save(using=self._db)
        return user
       
class User(AbstractBaseUser):
    email_error_message = 'Email must be: @plm.edu.ph'
    email_regex = RegexValidator(
    regex=r'^[A-Za-z0-9._%+-]+@plm.edu.ph$',
    message=email_error_message
    )
    # ID number code. Can be copy pasted to suit ID code for certain user.
    ID_error_message = 'Faculty ID must be entered in format: 20XXXXXXX'
    ID_regex = RegexValidator(
        regex=r'^20\d{7}$',
        message=ID_error_message
    )
    email = models.EmailField(verbose_name="Email",max_length=60,unique=True,validators=[email_regex])
    firstName = models.CharField(max_length=100, verbose_name='First Name', null=True,blank=True)
    lastName = models.CharField(max_length=100, verbose_name='Last Name', null=True,blank=True)
    numberID = models.IntegerField(validators=[ID_regex],unique=True, verbose_name='Number ID', null=True)
    is_active = models.BooleanField(default=True)
    date_of_inactive = models.DateTimeField(verbose_name="Date of Inacitve",null=True,blank=True)
    is_admin = models.BooleanField(default=False,verbose_name='Admin')
    has_answer = models.BooleanField(default=False,verbose_name='Has Answered')
    date_joined = models.DateTimeField(verbose_name='Date Joined',auto_now_add=True)
    last_login = models.DateTimeField(verbose_name="Last Login",null=True,blank=True)
 
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
 
    objects = UserManager()
 
    def __str__(self):
        return f"{self.email}"
 
    def has_perm(self, perm, obj=None):
        return True
 
    def has_module_perms(self, app_label):
        """Does the user have permissions to view the app `app_label`?"""
        # Simplest possible answer: Yes, always
        return True
 
    @property
    def is_staff(self):
        """Is the user a member of staff?"""
        # Simplest possible answer: All admins are staff
        return self.is_admin

class College(models.Model):
    college = models.CharField(max_length=100, verbose_name="College",unique=True,null=True, blank =True)
 
    def __str__(self):
        return f"{self.college}"
 
class Course(models.Model):
    college = models.ForeignKey(College, verbose_name='College',null=True,blank=True,on_delete=models.CASCADE)
    course = models.CharField(max_length=100, verbose_name="Course",unique=True,null=True, blank =True)
 
    def __str__(self):
        return f"{self.course}"

class Answers(models.Model):
    email_error_message = 'Email must be: @plm.edu.ph'
    email_regex = RegexValidator(
    regex=r'^[A-Za-z0-9._%+-]+@plm.edu.ph$',
    message=email_error_message
    )
    # ID number code. Can be copy pasted to suit ID code for certain user.
    ID_error_message = 'Faculty ID must be entered in format: 20XXXXXXX'
    ID_regex = RegexValidator(
        regex=r'^20\d{7}$',
        message=ID_error_message
    )
    Year_CHOICES = (
        ('1st Year', '1st Year'),
        ('2nd Year', '2nd Year'),
        ('3rd Year', '3rd Year'),
        ('4th Year', '4th Year'),
        ('5th Year', '5th Year'),
        ('6th Year', '6th Year'),
    )
    email = models.EmailField(verbose_name="Email",max_length=60,validators=[email_regex],blank=True)
    firstName = models.CharField(max_length=100, verbose_name='First Name', null=True,blank=True)
    lastName = models.CharField(max_length=100, verbose_name='Last Name', null=True,blank=True)
    numberID = models.CharField(max_length=100,validators=[ID_regex], verbose_name='Number ID', null=True,blank=True)
    college= models.ForeignKey(College, verbose_name='College',null=True,blank=True,on_delete=models.CASCADE)
    course= models.ForeignKey(Course, verbose_name='Course',null=True,blank=True,on_delete=models.CASCADE)
    year = models.CharField(max_length=10, choices=Year_CHOICES, verbose_name='Year', null=True,blank=True)
    block = models.CharField(max_length=10, verbose_name='Block', null=True,blank=True)
    question1 = models.CharField(max_length=1000, verbose_name="question1",null=True, blank =True)
    question2 = models.CharField(max_length=1000, verbose_name="question2",null=True, blank =True)
    question3 = models.CharField(max_length=1000, verbose_name="question3",null=True, blank =True)
    question4 = models.CharField(max_length=1000, verbose_name="question4",null=True, blank =True)
    question5 = models.CharField(max_length=1000, verbose_name="question5",null=True, blank =True)

    def __str__(self):
        return f"{self.numberID}"