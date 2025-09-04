from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .utils import extract_text_from_file, calculate_similarity, get_file_extension
from .models import UserProfile

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', 'home')
            return redirect(next_url)
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('home')

def home(request):
    context = {
        'username': request.user.username if request.user.is_authenticated else None
    }
    return render(request, 'home.html', context)

@login_required
def compare_documents(request):
    if request.method == 'POST':
        try:
            # Get uploaded files
            file1 = request.FILES.get('file1')
            file2 = request.FILES.get('file2')

            if not file1 or not file2:
                return render(request, 'compare.html', {
                    'error': 'Please upload both files for comparison.'
                })

            # Get file extensions
            ext1 = get_file_extension(file1.name)
            ext2 = get_file_extension(file2.name)

            # Extract text from files
            text1 = extract_text_from_file(file1, ext1)
            text2 = extract_text_from_file(file2, ext2)

            # Calculate similarity
            similarity_percentage = calculate_similarity(text1, text2)

            # Prepare result data
            result = {
                'file1_name': file1.name,
                'file2_name': file2.name,
                'similarity_percentage': similarity_percentage,
                'text1_preview': text1[:500] + '...' if len(text1) > 500 else text1,
                'text2_preview': text2[:500] + '...' if len(text2) > 500 else text2,
            }

            return render(request, 'compare.html', {
                'result': result,
                'success': True
            })

        except Exception as e:
            error_message = f"Error processing files: {str(e)}"
            return render(request, 'compare.html', {
                'error': error_message
            })

    return render(request, 'compare.html')

@login_required
def profile(request):
    return render(request, 'profile.html')