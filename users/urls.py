# users/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page for login and signup
    path('health_data/', views.health_data, name='health_data'),  # Health data input page
    path('visualization/', views.data_visualization, name='data_visualization'),  # Data visualization page
    path('logout/', views.logout_view, name='logout'),  # Logout functionality
]
