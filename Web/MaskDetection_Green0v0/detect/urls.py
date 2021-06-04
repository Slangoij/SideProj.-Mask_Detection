# 2차 url
from django.urls import path
# from .views import *
from . import views

# 이 url pattern들의 namespace(prefix)로 사용할 값 설정
# urlpattern 설정의 이름 호출 시 다른 app들과 구분하기 위해 사용한다.
app_name = "detect"

urlpatterns = [
    path("",views.home, name='home'),
    path("test/", views.test, name='test'),
    path('temp/', views.temp, name='temp'),
    # path("cantoimg/", views.canvasToImage, name='cantoimg'),
]