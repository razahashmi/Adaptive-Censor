"""Vectorized equivalent of NudeNet 3.4.2's class-agnostic postprocessing."""
import cv2
import numpy as np

LABELS = (
    'FEMALE_GENITALIA_COVERED','FACE_FEMALE','BUTTOCKS_EXPOSED','FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED','MALE_BREAST_EXPOSED','ANUS_EXPOSED','FEET_EXPOSED',
    'BELLY_COVERED','FEET_COVERED','ARMPITS_COVERED','ARMPITS_EXPOSED','FACE_MALE',
    'BELLY_EXPOSED','MALE_GENITALIA_EXPOSED','ANUS_COVERED','FEMALE_BREAST_COVERED','BUTTOCKS_COVERED',
)


def postprocess(output, x_pad, y_pad, x_ratio, y_ratio, image_original_width,
                image_original_height, model_width, model_height, include_class_scores=False):
    rows=np.transpose(np.squeeze(output[0]))
    scores=rows[:,4:].max(axis=1)
    keep=scores>=.2
    if not keep.any():return []
    rows=rows[keep];scores=scores[keep]
    classes=rows[:,4:].argmax(axis=1)
    # Preserve the reference arithmetic order, clipping and class-agnostic NMS.
    x=(rows[:,0]-rows[:,2]/2)*(image_original_width+x_pad)/model_width
    y=(rows[:,1]-rows[:,3]/2)*(image_original_height+y_pad)/model_height
    w=rows[:,2]*(image_original_width+x_pad)/model_width
    h=rows[:,3]*(image_original_height+y_pad)/model_height
    x=np.clip(x,0,image_original_width);y=np.clip(y,0,image_original_height)
    w=np.minimum(w,image_original_width-x);h=np.minimum(h,image_original_height-y)
    boxes=np.stack([x,y,w,h],axis=1)
    indices=cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),.25,.45)
    result=[dict(**{'class':LABELS[classes[i]]},score=float(scores[i]),
                 box=[int(v) for v in boxes[i]]) for i in indices]
    if include_class_scores:
        for hit,i in zip(result,indices):
            hit['class_scores']=rows[i,4:].tolist()
    return result


def detect(model, image, include_class_scores=False):
    from nudenet.nudenet import _read_image
    blob,x_ratio,y_ratio,x_pad,y_pad,width,height=_read_image(image,model.input_width)
    output=model.onnx_session.run(None,{model.input_name:blob})
    return postprocess(output,x_pad,y_pad,x_ratio,y_ratio,width,height,model.input_width,model.input_height,
                       include_class_scores=include_class_scores)
