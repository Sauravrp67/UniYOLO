import sys
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import YOLOv3_DPU,YOLOv3,BoxDecoder,do_sigmoid
from dataloader import Dataset,BasicTransform,AugmentTransform
from torch.utils.data import DataLoader,Subset
from utils import YOLOv3Loss,scale_coords,transform_xcycwh_to_x1y1x2y2,transform_x1y1x2y2_to_x1y1wh,filter_confidence,run_NMS,to_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def validate(args,anchors, dataloader, model, evaluator, dpu:bool = False,save_result=False,save_filename:str = "predictions.txt"):
    model.eval()
    if not dpu :
        model.module.set_grid_xy(input_size=args.img_size) if hasattr(model, "module") else model.set_grid_xy(input_size=args.img_size)

    with open(args.mAP_filepath, mode="r") as f:
        mAP_json = json.load(f)

    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    imageToid = mAP_json["imageToid"]

    for _, minibatch in enumerate(dataloader):
        filenames, images, shapes = minibatch[0], minibatch[1], minibatch[3]
        predictions = model(images.to("cpu"))
        if dpu:
            decoded_52 = BoxDecoder(predictions[0],torch.tensor(anchors[0:3])).decode_predictions()
            decoded_26 = BoxDecoder(predictions[1],torch.tensor(anchors[3:6])).decode_predictions()
            decoded_13 = BoxDecoder(predictions[2],torch.tensor(anchors[6:9])).decode_predictions()
            predictions = torch.cat((decoded_52,decoded_26,decoded_13),dim = 1)
        # cuda(args.rank, non_blocking=True)

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()

            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
            prediction = filter_confidence(prediction=prediction, conf_threshold=args.conf_thres)
            prediction = run_NMS(prediction=prediction, iou_threshold=args.nms_thres)

            if len(check_images) < 5:
                check_images.append(to_image(images[j]))
                check_preds.append(prediction.copy())
                
            if len(prediction) > 0:
                filename = filenames[j]
                shape = shapes[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1x2y2 = scale_coords(img1_shape=images.shape[2:], coords=prediction[:, 1:5], img0_shape=shape[:2])
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=box_x1y1x2y2)
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    del images, predictions
    torch.cuda.empty_cache()

    if len(cocoPred) > 0:
        cocoPred = np.concatenate(cocoPred, axis=0)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)

        if save_result:
            np.savetxt(args.exp_path / save_filename, cocoPred, fmt="%.4f", delimiter=",", header=f"Inference results of [image_id, x1y1wh, score, label]") 
            out_file = args.exp_path / save_filename
            with open(out_file, "a") as f:
                f.write(eval_text)
        return mAP_dict, eval_text
    else:
        return None, None

def get_dataloader(voc_path,batch_size,subset_length = 8,train = False,input_size = 416):
    k = subset_length
    if not train:
        val_dataset = Dataset(voc_path,phase = 'val')
        print(len(val_dataset))
        subset_index = list(range(min(k,len(val_dataset))))
        val_transform = BasicTransform(input_size = input_size)
        val_dataset.load_transformer(val_transform)
        val_subset = Subset(val_dataset,indices = subset_index)
        return DataLoader(dataset=val_subset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1,drop_last=True)
    
    else:
        train_dataset = Dataset(voc_path,phase = "train")
        subset_index = list(range(min(k,len(train_dataset))))
        train_transform = AugmentTransform(input_size = input_size,dataset = train_dataset)
        train_dataset.load_transformer(train_transform)
        train_subset = Subset(train_dataset,indices = subset_index)
        return DataLoader(dataset=train_subset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1,drop_last=True)
    

def load_model(mode:str,device: str,input_size: int, num_classes: int,model_type: str,anchors,model_path:str):
    """
    Initialize YOLO model and load checkpoint.
    mode:"normal" -> uninspected model,some layers might not be supported in the dpu
          "dpu" -> dpu supported model
          "dpu_quantized" -> Quantized Model in DPU.
         
    """
    if mode == "dpu":
        model = YOLOv3_DPU(input_size = input_size, num_classes = num_classes, anchors = anchors, model_type = model_type, pretrained = False).to(device)
        ckpt = torch.load(model_path,map_location = 'cpu',weights_only = False)
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing,unexpected = model.load_state_dict(sd,strict = True)
        model.set_grid_xy(input_size = input_size)
        print(f"[load] missing = {len(missing)} unexpected = {len(unexpected)}")
    
    elif mode == "normal":
        model = YOLOv3(input_size = input_size, num_classes = num_classes, anchors = anchors, model_type = model_type, pretrained = False).to(device)
        ckpt = torch.load(model_path,map_location = 'cpu',weights_only = False)
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing,unexpected = model.load_state_dict(sd,strict = True)
        model.set_grid_xy(input_size = input_size)
        print(f"[load] missing = {len(missing)} unexpected = {len(unexpected)}")
    elif mode == "dpu_quantized":
        model = torch.jit.load(model_path,map_location=device)
    
    model.to(device).eval()
    model.zero_grad() 
    return model


def evaluate(model,val_loader,criterion):
    losses = []
    with torch.no_grad():
        model.eval()
        for _,images,targets,_ in tqdm(val_loader,):
            inputs = images.to(device)
            targets = targets
            outputs = model(inputs)
            
            preds0 = do_sigmoid(outputs[0])
            preds1 = do_sigmoid(outputs[1])
            preds2 = do_sigmoid(outputs[2])
            outputs = (preds0,preds1,preds2)
            loss = criterion(outputs,targets)
            print(loss)
            losses.append(loss[0])
    avg_loss = sum(losses) / len(losses)

    print(f"Avg_loss:{avg_loss}")

    return avg_loss