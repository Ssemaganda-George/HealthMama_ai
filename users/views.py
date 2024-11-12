# users/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import HealthMetrics
import matplotlib.pyplot as plt
import plotly.express as px
import os
from django.conf import settings

def home(request):
    if request.user.is_authenticated:
        return redirect('health_data')

    if request.method == 'POST':
        if 'login' in request.POST:
            # Login form
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('health_data')
            else:
                messages.error(request, 'Invalid username or password.')
        
        elif 'signup' in request.POST:
            # Signup form with basic validation
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists.')
            else:
                User.objects.create_user(username=username, email=email, password=password)
                messages.success(request, 'Account created successfully. You can now log in.')

    return render(request, 'users/home.html')


# Data input view
@login_required
def health_data(request):
    if request.method == 'POST':
        blood_pressure = request.POST['blood_pressure']
        glucose_level = request.POST['glucose_level']
        health_entry = HealthMetrics(user=request.user, blood_pressure=blood_pressure, glucose_level=glucose_level)
        health_entry.save()
        return redirect('health_data')
    
    # data = HealthMetrics.objects.filter(user=request.user).order_by('-timestamp')
    data = HealthMetrics.objects.all()
    return render(request, 'users/health_data.html', {'data': data})



# Data visualization page
@login_required
def data_visualization(request):
    data = HealthMetrics.objects.filter(user=request.user).order_by('-timestamp')
    
    if not data:
        messages.warning(request, 'No data available to display.')
        return redirect('health_data')

    # Prepare data for plotting
    blood_pressure_data = [entry.blood_pressure for entry in data]
    glucose_level_data = [entry.glucose_level for entry in data]
    
    # Matplotlib plot
    fig, ax = plt.subplots()
    ax.plot(blood_pressure_data, label='Blood Pressure')
    ax.plot(glucose_level_data, label='Glucose Level')
    ax.set_title('Health Data over Time')
    ax.legend()

    # Save plot to a dynamic path
    plot_path = os.path.join(settings.MEDIA_ROOT, 'plot.png')
    fig.savefig(plot_path)

    # Plotly interactive chart
    fig_plotly = px.line(x=list(range(len(blood_pressure_data))), y=blood_pressure_data, title='Blood Pressure Trend')
    plotly_html = fig_plotly.to_html(full_html=False)

    return render(request, 'users/visualization.html', {
        'plot_path': plot_path,
        'plotly_html': plotly_html
    })

def logout_view(request):
    logout(request)
    return redirect('home')
