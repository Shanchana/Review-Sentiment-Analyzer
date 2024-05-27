
from django.urls import path
from . import views


urlpatterns =[
    path('',views.link , name="link"),
    path('output',views.output , name="output")
]