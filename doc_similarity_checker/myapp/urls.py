from django.urls import path
from . import views

urlpatterns = [
    path('auth/register/', views.register, name='register'),
    path('auth/login/', views.user_login, name='login'),
    path('', views.home, name='home'),
    path('compare/', views.compare_documents, name='compare_documents'),
    path('profile/', views.profile, name='profile'),
    path('logout/', views.user_logout, name='logout'),
]