# Generated by Django 4.1.4 on 2022-12-09 16:06

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('email', models.EmailField(max_length=60, unique=True, validators=[django.core.validators.RegexValidator(message='Email must be: @plm.edu.ph', regex='^[A-Za-z0-9._%+-]+@plm.edu.ph$')], verbose_name='Email')),
                ('firstName', models.CharField(blank=True, max_length=100, null=True, verbose_name='First Name')),
                ('lastName', models.CharField(blank=True, max_length=100, null=True, verbose_name='Last Name')),
                ('numberID', models.IntegerField(null=True, unique=True, validators=[django.core.validators.RegexValidator(message='Faculty ID must be entered in format: 20XXXXXXX', regex='^20\\d{7}$')], verbose_name='Number ID')),
                ('is_active', models.BooleanField(default=True)),
                ('date_of_inactive', models.DateTimeField(blank=True, null=True, verbose_name='Date of Inacitve')),
                ('is_admin', models.BooleanField(default=False, verbose_name='Admin')),
                ('has_answer', models.BooleanField(default=False, verbose_name='Has Answered')),
                ('date_joined', models.DateTimeField(auto_now_add=True, verbose_name='Date Joined')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='Last Login')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='College',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('college', models.CharField(blank=True, max_length=100, null=True, unique=True, verbose_name='College')),
            ],
        ),
        migrations.CreateModel(
            name='Course',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('course', models.CharField(blank=True, max_length=100, null=True, unique=True, verbose_name='Course')),
                ('college', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='capswebsite.college', verbose_name='College')),
            ],
        ),
        migrations.CreateModel(
            name='Answers',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(blank=True, max_length=60, validators=[django.core.validators.RegexValidator(message='Email must be: @plm.edu.ph', regex='^[A-Za-z0-9._%+-]+@plm.edu.ph$')], verbose_name='Email')),
                ('firstName', models.CharField(blank=True, max_length=100, null=True, verbose_name='First Name')),
                ('lastName', models.CharField(blank=True, max_length=100, null=True, verbose_name='Last Name')),
                ('numberID', models.CharField(blank=True, max_length=100, null=True, validators=[django.core.validators.RegexValidator(message='Faculty ID must be entered in format: 20XXXXXXX', regex='^20\\d{7}$')], verbose_name='Number ID')),
                ('year', models.CharField(blank=True, choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], max_length=10, null=True, verbose_name='Year')),
                ('block', models.CharField(blank=True, max_length=10, null=True, verbose_name='Block')),
                ('question1', models.CharField(blank=True, max_length=1000, null=True, verbose_name='question1')),
                ('question2', models.CharField(blank=True, max_length=1000, null=True, verbose_name='question2')),
                ('question3', models.CharField(blank=True, max_length=1000, null=True, verbose_name='question3')),
                ('question4', models.CharField(blank=True, max_length=1000, null=True, verbose_name='question4')),
                ('question5', models.CharField(blank=True, max_length=1000, null=True, verbose_name='question5')),
                ('college', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='capswebsite.college', verbose_name='College')),
                ('course', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='capswebsite.course', verbose_name='Course')),
            ],
        ),
    ]
