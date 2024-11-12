# mama_ai/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # User-related routes (handles home, login, register, health data, etc.)
    path('', include('users.urls')),  # Root URL for home and other user views
    
    # AI-related routes
    path('ai/', include('ai.urls')),  # AI app's routes, including chat
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
