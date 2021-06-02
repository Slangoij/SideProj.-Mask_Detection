# detect/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse # reverse함수 : path 설정 이름으로 url 문자열을 만들어 주는 함수

# Create your views here.
# http://127.0.0.1/detect/test
def home(request):
    html = "<html><body>Hi Django</body></html>"
    return HttpResponse(html)
    
def test(request):
    print('test뷰 통과 완료')
    return render(request, "detect/test.html")
