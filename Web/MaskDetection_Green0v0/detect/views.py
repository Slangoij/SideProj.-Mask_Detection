# detect/views.py
from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse # reverse함수 : path 설정 이름으로 url 문자열을 만들어 주는 함수

from django.views.decorators.csrf import csrf_exempt
import random
import os
from django.conf import settings
import base64

# Create your views here.
# http://127.0.0.1/detect/test
def home(request):
    html = "<html><body>Hi Django</body></html>"
    return HttpResponse(html)

def test(request):
    print('test뷰 통과 완료')
    return render(request, "detect/test.html")

@csrf_exempt
def temp(request):
    data = request.POST.__getitem__('data')
    # print(data)
    data = data[22:]  # 앞의 'data:image/png;base64' 부분 제거
    number = random.randrange(1,10000) # 동시에 다른 사용자가 접근시 최대한 중복을 막기

    # 저장할 경로 및 파일명을 지정
    path = str(os.path.join(settings.STATIC_ROOT, 'image'))
    filename = 'image' + str(number) + '.png'

    # 바이너리 쓰기'wb'모드로 파일 open
    image = open(path + filename, 'wb')
    # 'base64.b54decode()'메소드로 파일 디코딩 후 파일쓰기
    image.write(base64.b64decode(data))
    image.close()

    # filename을 json형식에 맞추어 response를 보내기
    answer = {'filename':filename}
    print('디코딩 완료')
    # return render(request, "detect/test.html")
    return JsonResponse(answer)


# # canvas 이미지 저장
# @csrf_exempt
# def canvasToImage(request):
#     data = request.POST.__getitem__('data')
#     data = data[22:]  # 앞의 'data:image/png;base64' 부분 제거
#     number = random.rand(1,10000)

#     # 저장할 경로 및 파일명을 지정
#     path = str(os.path.join(settings.STATIC_ROOT, 'image/'))
#     filename = 'image' + str(number) + '.png'

#     # 바이너리 쓰기'wb'모드로 파일 open
#     image = open(path + filename, 'wb')
#     # 'base64.b54decode()'메소드로 파일 디코딩 후 파일쓰기
#     image.write(base64.b64decode(data))
#     image.close()

#     # filename을 json형식에 맞추어 response를 보내기
#     answer = {'filename':filename}
#     print('디코딩 완료')
#     return JsonResponse(answer)