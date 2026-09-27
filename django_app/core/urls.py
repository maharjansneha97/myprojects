from django.urls import path
from . import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('history', views.history, name='history'),
    path('history/clear', views.clear_history, name='clear_history'),
    path('history/download/<int:analysis_id>', views.download_analysis, name='download_analysis'),
    path('signup', views.signup, name='signup'),
    path('account/delete', views.delete_account, name='delete_account'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('upload', views.upload_image, name='upload_image'),
    path('api/status', views.status, name='status'),
]
