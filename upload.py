import concurrent.futures
from firebase_admin import storage
from firebase_admin import credentials
import firebase_admin
import uuid
from io import BytesIO

cred = credentials.Certificate('/ai-studio-credential.json')
app = firebase_admin.initialize_app(cred, {
  "apiKey": "AIzaSyDHzXueVeh_I4m5kemyCVGmYtu_uDfQ1Mw",
  "authDomain": "ai-art-app-32bd3.firebaseapp.com",
  "projectId": "ai-art-app-32bd3",
  "storageBucket": "ai-art-app-32bd3.appspot.com",
  "messagingSenderId": "968016255589",
  "appId": "1:968016255589:web:443264566a633de98f448c",
  "measurementId": "G-Z1JTTFC6WV",
})

bucket = storage.bucket(app=app)

def upload_image(image, prompt, modifiers):

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_id = str(uuid.uuid4())
    ref = "aistudio/"+image_id+".png"
    buffered.seek(0)
    blob = bucket.blob(ref)
    blob.metadata={
        "prompt":prompt,
        "modifiers": modifiers
    }
    blob.upload_from_file(buffered, content_type='image/png')
    blob.make_public()
    return blob.public_url


