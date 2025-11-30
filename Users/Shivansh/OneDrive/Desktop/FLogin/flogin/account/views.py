from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile

import base64
import cv2
import numpy as np

from django.contrib.auth.models import User
from .models import UserImages


def _load_face_cascade():
    model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(model_path)
    return face_cascade


@csrf_exempt
def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        face_image_data = request.POST.get('face_image')

        if not username or not face_image_data:
            return JsonResponse({
                'status': 'error',
                'message': 'There is no face image or username.'
            })

        try:
            base64_str = face_image_data.split(",")[1]
            img_bytes = base64.b64decode(base64_str)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'The image could not be read. Please capture again.'
                })

        except Exception:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid image format. (Base64 decode fail)'
            })

        face_cascade = _load_face_cascade()

        if face_cascade is None or face_cascade.empty():
            return JsonResponse({
                'status': 'error',
                'message': 'Face detection model could not be loaded.'
            })

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(120, 120)
        )

        if len(faces) == 0:
            return JsonResponse({
                'status': 'error',
                'message': 'No face detected. Move closer to the camera and look straight ahead.'
            })

        if len(faces) > 1:
            return JsonResponse({
                'status': 'error',
                'message': 'Multiple faces detected. Only one person should be in front of the camera.'
            })

        (x, y, w, h) = faces[0]

        img_h, img_w = gray.shape
        face_area = w * h
        frame_area = img_w * img_h

        if face_area < 0.10 * frame_area:
            return JsonResponse({
                'status': 'error',
                'message': 'The face is too distant. Get closer to the camera and give it another go.'
            })

        if User.objects.filter(username=username).exists():
            return JsonResponse({
                'status': 'error',
                'message': 'This user name is already registered. Please choose another one.'
            })

        face_roi = img[y:y + h, x:x + w]

        success, buffer = cv2.imencode('.jpg', face_roi)
        if not success:
            return JsonResponse({
                'status': 'error',
                'message': 'Face did not save properly. Please try again.'
            })

        face_bytes = buffer.tobytes()
        face_image_file = ContentFile(face_bytes, name=f"{username}_face.jpg")

        user = User.objects.create(username=username)

        user_image = UserImages.objects.create(
            user=user,
            face_image=face_image_file
        )

        login_id = f"U{user_image.id:04d}"
        user_image.login_id = login_id
        user_image.save(update_fields=['login_id'])

        return JsonResponse({
            'status': 'success',
            'message': f'Your Photo is successful Capture! Your Login ID is {login_id}. Please copy your Login ID for future Logins.',
            'login_id': login_id,
            'redirect': '/login/'
        })

    return render(request, 'register.html')


@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        login_id = request.POST.get('login_id')
        face_image_data = request.POST.get('face_image')

        if not username or not login_id or not face_image_data:
            return JsonResponse({
                'status': 'error',
                'message': 'Username, Login ID or face image is missing.'
            })

        try:
            user_face_obj = UserImages.objects.select_related('user').get(login_id=login_id)
        except UserImages.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid Login ID. Please enter correct ID.'
            })

        user = user_face_obj.user

        if user.username != username:
            return JsonResponse({
                'status': 'error',
                'message': 'Username and Login ID do not match.'
            })

        try:
            base64_str = face_image_data.split(",")[1]
            img_bytes = base64.b64decode(base64_str)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Image did not read. Try again.'
                })

        except Exception:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid image format (login).'
            })

        face_cascade = _load_face_cascade()
        if face_cascade is None or face_cascade.empty():
            return JsonResponse({
                'status': 'error',
                'message': 'Face model did not load (login).'
            })

        gray_login = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_login,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(120, 120)
        )

        if len(faces) == 0:
            return JsonResponse({
                'status': 'error',
                'message': 'Did not detect a face. Please come nearer.'
            })

        if len(faces) > 1:
            return JsonResponse({
                'status': 'error',
                'message': 'Please keep only one person in front of the camera during login.'
            })

        (x, y, w, h) = faces[0]
        login_face = gray_login[y:y + h, x:x + w]

        stored_face_path = user_face_obj.face_image.path
        stored_img = cv2.imread(stored_face_path, cv2.IMREAD_GRAYSCALE)

        if stored_img is None:
            return JsonResponse({
                'status': 'error',
                'message': 'The stored face image could not be loaded.'
            })

        target_size = (200, 200)
        login_face_resized = cv2.resize(login_face, target_size)
        stored_face_resized = cv2.resize(stored_img, target_size)

        hist_login = cv2.calcHist([login_face_resized], [0], None, [256], [0, 256])
        hist_stored = cv2.calcHist([stored_face_resized], [0], None, [256], [0, 256])

        cv2.normalize(hist_login, hist_login)
        cv2.normalize(hist_stored, hist_stored)

        similarity = cv2.compareHist(hist_login, hist_stored, cv2.HISTCMP_CORREL)
        THRESHOLD = 0.7

        if similarity >= THRESHOLD:
            return JsonResponse({
                'status': 'success',
                'message': f'Login successful! Welcome {username}.',
                'redirect': '/dashboard/'
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'The faces did not match. Look straight into the camera and try again in good light.'
            })

    return render(request, 'login.html')


@csrf_exempt
def delete_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        login_id = request.POST.get('login_id')

        if not username or not login_id:
            return JsonResponse({
                'status': 'error',
                'message': 'Both username and login ID are required.'
            })

        try:
            user_face_obj = UserImages.objects.select_related('user').get(login_id=login_id)
        except UserImages.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid login ID. No users found..'
            })

        user = user_face_obj.user

        if user.username != username:
            return JsonResponse({
                'status': 'error',
                'message': 'Username and login ID do not match. Deletion not allowed'
            })

        if user_face_obj.face_image:
            user_face_obj.face_image.delete(save=False)

        user.delete()

        return JsonResponse({
            'status': 'success',
            'message': f'User "{username}" (ID: {login_id}) and your face are successfully deleted from the system'
        })

    return render(request, 'delete.html')


def dashboard(request):
    username = request.GET.get('username', '')
    login_id = request.GET.get('login_id', '')

    context = {
        'username': username,
        'login_id': login_id,
    }
    return render(request, 'dashboard.html', context)
