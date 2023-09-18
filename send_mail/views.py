from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from rest_framework.views import APIView
from deepface import DeepFace
import cv2
import os
import whisper
from .models import EmotionTrack
from datetime import datetime, timedelta
# from speechbrain.pretrained.interfaces import foreign_class

class facialExpressionAPI(APIView):
    def post(self, request):
        check_img = request.FILES.get("image", 0)
        uui = request.POST.get("uuid", 0)
        print(uui)
        if check_img == 0:
            return Response("provide an image")
        cwd = os.path.join('.',"child."+check_img.name.split('.')[-1])  
        with open (cwd, "wb") as f:
            f.write(check_img.read())
        analyze_face = DeepFace.analyze(cv2.imread("child.jpg"),detector_backend = 'retinaface', actions = ['emotion'])
        os.remove("child.jpg")
        if analyze_face[0]["dominant_emotion"] == 'neutral' and analyze_face[0]['emotion']['sad']>=10 and analyze_face[0]['emotion']['sad']<=23:
            analyze_face[0]["dominant_emotion"] = 'confusion'
        try:
            data = EmotionTrack(emotion = analyze_face[0]["dominant_emotion"], uuid =uui, time = datetime.now())
            data.save()
        except Exception as e:
            print(e)
        print(analyze_face)

        return Response(analyze_face)
    
#confusion, happy, sad, angry, neutral
#
from rest_framework import serializers
class EmotionTrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmotionTrack
        fields = ['uuid','emotion','time'] 

class getDataAPI(APIView):
    def post(self, request):
        current = datetime.now()
        result = current - timedelta(seconds=15)
        d = [i for i in EmotionTrack.objects.all() if i.time >= result.time()]
        data = EmotionTrackSerializer(d, many=True).data
        return Response(data)
    
class transcriptAPI(APIView):
    def post(slef, request):
        voice_model = whisper.load_model("large-v2")
        transcript_audio = request.FILES.get("image", 0)
        if transcript_audio == 0:
            return Response("provide an audio")
        with open("audio.webm", "wb") as f:
            f.write(transcript_audio.read())
        result = voice_model.transcribe(whisper.pad_or_trim(whisper.load_audio("audio.webm")))["text"]
        os.remove("audio.webm")
        return Response({"text": result})

# class speechEmotion(APIView):
#     def post(self, request):
        # classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        # out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
        # print(text_lab)