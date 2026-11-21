"""Download the models the fan-edit tools use into $FAN_EDIT_WORK/models (default ~/fan-edits/models).
YuNet (faces) and SFace (who is who) come from the OpenCV Hugging Face mirrors; U^2-Net human segmentation
and Real-ESRGAN come from their GitHub releases."""
import os, urllib.request
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))
M = os.path.join(WORK, 'models'); os.makedirs(M, exist_ok=True)
FILES = {
    'yunet.onnx': 'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx',
    'sface.onnx': 'https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx',
    'u2net_human_seg.onnx': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx',
    'realesr-general-x4v3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    'realesr-general-wdn-x4v3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
}
for name, url in FILES.items():
    p = os.path.join(M, name)
    if os.path.exists(p) and os.path.getsize(p) > 100000: print('have', name); continue
    print('get ', name); urllib.request.urlretrieve(url, p)
print('models in', M)
