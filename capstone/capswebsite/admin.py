from django.contrib import admin
from django.contrib.auth.models import Group
from django.contrib.auth.admin import UserAdmin
from .models import *
 
 
# Register your models here.
class UserAdmin(UserAdmin):
    list_display = ('pk','email','numberID','date_of_inactive','is_active','is_admin','date_joined','last_login')
    search_fields = ('email','numberID',)
    readonly_fields = ('date_joined','last_login')
    ordering = ('pk',)
    filter_horizontal = ()
    list_filter = ('is_active',)
    fieldsets = (
        (None, {'fields': ('email', 'password','numberID')}),
        ('Permissions', {'fields': ('is_active','date_of_inactive','is_admin')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email','numberID', 'password1', 'password2','is_admin'),
        }),
    )
 
 
class CollegeAdmin(admin.ModelAdmin):
    list_display = ('pk','college')
    search_fields = ('pk','college',)
    ordering = ('pk',)
    filter_horizontal = ()
    list_filter = ('college',)

class CourseAdmin(admin.ModelAdmin):
    list_display = ('pk','college','course')
    search_fields = ('pk','college','course')
    ordering = ('pk',)
    filter_horizontal = ()
    list_filter = ('college',)

class AnswersAdmin(admin.ModelAdmin):
    list_display = ('pk',)
    search_fields = ('pk',)
    ordering = ('pk',)
    filter_horizontal = ()


admin.site.register(User,UserAdmin)
admin.site.register(College,CollegeAdmin)
admin.site.register(Course,CourseAdmin)
admin.site.register(Answers,AnswersAdmin)
admin.site.unregister(Group)