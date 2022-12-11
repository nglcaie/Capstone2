from django.urls import include, path, re_path
from django.conf import settings
from . import views
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('login',views.loginPage, name="login"),
     path('logout', views.logout_view, name="logout"),
    path('sign_up',views.sign_upPage, name="sign_up"),
    path('evaluation',views.evaluation, name="evaluation"),
    path('student_list',views.student_list, name="student_list"),
    #path('student_answer',views.student_answer, name="student_answer"),
    re_path(r'^student_answer/(?P<pk>[a-zA-Z0-9]+$)', views.student_answer, name='student_answer'),
    path('answer_summary',views.answer_summary, name="answer_summary"),
    path('start_survey',views.start_survey, name="start_survey"),
    path('survey_question',views.survey_question, name="survey_question"),
    path('student_navbar',views.student_navbar, name="student_navbar"),
    path('thankyou',views.thankyou, name="thankyou"),
    path('admin_navbar',views.admin_navbar, name="admin_navbar"),
    path('ajax/load-slot/', views.load_slot, name='ajax_load_slot'),
    path('answers_csv',views.answers_csv, name="answers_csv"),
    path('word_cloud_page',views.word_cloud_page, name="word_cloud_page"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)