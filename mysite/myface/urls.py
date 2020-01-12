from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('my-ajax-test/', views.testcall, name='ajax-test-view')
]