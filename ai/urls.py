# ai/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),  # Map the chat view to '/chat/'
]
