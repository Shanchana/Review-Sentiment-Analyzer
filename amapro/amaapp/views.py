from django.shortcuts import render
from .utils import main_process

# Create your views here.

def link(request):
  return render (request ,'link.html')

def output(request):
    if request.method == 'POST':
       url = request.POST["url"]
       results = main_process(url)
       return render(request, 'output.html' , {'results': results})
    return render (request ,'output.html')