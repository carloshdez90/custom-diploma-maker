from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from google.cloud import storage
from PIL import Image
import numpy as np
import datetime
import uuid
import sys
import cv2
import os
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './bucket-service-account.json'
DIPLOMA_TEMPLATE="tpl01.jpg"
SOURCE_TEMPLATE="input/source01.jpg"
OUTPUT_FOLDER="output/"
BUCKET="diploma-maker"

app = FastAPI()

def get_image(bucket_name, image):
  """Get file form bucket"""
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(image)

  image = np.asarray(bytearray(blob.download_as_string()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
  image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
  return image

def upload_diploma(diploma):
  storage_client = storage.Client()
  bucket = storage_client.bucket(BUCKET)
  blob = bucket.blob(OUTPUT_FOLDER+str(uuid.uuid4())+'.png')
  
  bs = io.BytesIO()
  diploma.save(bs, format='PNG')
  bs = bs.getvalue()

  blob.upload_from_string(bs, content_type="image/png")

  return blob

def generate_download_signed_url_v4(blob):
    """Generates a v4 signed URL for downloading a blob."""
    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 10 minutes
        expiration=datetime.timedelta(minutes=10),
        # Allow GET requests using this URL.
        method="GET",
    )

    return url
		
def generate_diploma(diploma_image, source_image):
  print("[INFO] laoding files ...")

  image = get_image(BUCKET, diploma_image)
  (imgH, imgW) = image.shape[:2]

  # load the source image from disk
  source = get_image(BUCKET, source_image)
  # load the ArUCo dictionary, grab the ArUCo parameters, and detect
  # the markers
  print("[INFO] detecting markers...")
  arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
  arucoParams = cv2.aruco.DetectorParameters_create()
  (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
    parameters=arucoParams)

  # if we have not found four markers in the input image then we cannot
  # apply our augmented reality technique
  if len(corners) != 4:
    print("[INFO] could not find 4 corners...exiting")
    sys.exit(0)

  # otherwise, we've found the four ArUco markers, so we can continue
  # by flattening the ArUco IDs list and initializing our list of
  # reference points
  print("[INFO] constructing augmented reality visualization...")
  ids = ids.flatten()
  refPts = []

  # loop over the IDs of the ArUco markers in top-left, top-right,
  # bottom-right, and bottom-left order
  for i in (923, 1001, 241, 1007):
    # grab the index of the corner with the current ID and append the
    # corner (x, y)-coordinates to our list of reference points
    j = np.squeeze(np.where(ids == i))
    corner = np.squeeze(corners[j])
    refPts.append(corner)

  # unpack our ArUco reference points and use the reference points to
  # define the *destination* transform matrix, making sure the points
  # are specified in top-left, top-right, bottom-right, and bottom-left
  # order
  (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
  dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
  dstMat = np.array(dstMat)

  # grab the spatial dimensions of the source image and define the
  # transform matrix for the *source* image in top-left, top-right,
  # bottom-right, and bottom-left order
  (srcH, srcW) = source.shape[:2]
  srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

  # compute the homography matrix and then warp the source image to the
  # destination based on the homography
  (H, _) = cv2.findHomography(srcMat, dstMat)
  warped = cv2.warpPerspective(source, H, (imgW, imgH))

  # construct a mask for the source image now that the perspective warp
  # has taken place (we'll need this mask to copy the source image into
  # the destination)
  mask = np.zeros((imgH, imgW), dtype="uint8")
  cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
    cv2.LINE_AA)

  # this step is optional, but to give the source image a black border
  # surrounding it when applied to the source image, you can apply a
  # dilation operation
  rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  mask = cv2.dilate(mask, rect, iterations=2)

  # create a three channel version of the mask by stacking it depth-wise,
  # such that we can copy the warped source image into the input image
  maskScaled = mask.copy() / 255.0
  maskScaled = np.dstack([maskScaled] * 3)

  # copy the warped source image into the input image by (1) multiplying
  # the warped image and masked together, (2) multiplying the original
  # input image with the mask (giving more weight to the input where
  # there *ARE NOT* masked pixels), and (3) adding the resulting
  # multiplications together
  warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
  imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
  output = cv2.add(warpedMultiplied, imageMultiplied)
  output = output.astype("uint8")

  output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
  output = Image.fromarray(output)

  return output

@app.get("/", status_code=200)
def get_diploma(token: str =  None, source: str = None):
  if token is None:
    resp = JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"Error": "Invalid token"}) 
    return resp
    # TODO implement token validation with SSO
    
  if source is None:
    source = SOURCE_TEMPLATE

  try:
    diploma = generate_diploma(DIPLOMA_TEMPLATE, source)
    diploma = upload_diploma(diploma)

    #temporal link -> 10 minutes
    public_diploma_url = generate_download_signed_url_v4(diploma)
    
    resp = JSONResponse(content={"url": public_diploma_url})

  except  Exception as e:
    print(e)
    resp = JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"Error": "Invalid source image"}) 
    return resp

  return resp