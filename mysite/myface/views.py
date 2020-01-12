from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64

def index(request):
    print("yeet711")
    context = {}
    return render(request, 'myface/index.html', context)

@csrf_exempt
def testcall(request):
    imgdata = base64.b64decode(request.POST['img'].replace('data:image/png;base64,',''))

    filename = 'webScreenshot.jpg'

    with open(filename, 'wb') as f:
        f.write(imgdata)

    return HttpResponse("You're a wizard Harry!")
