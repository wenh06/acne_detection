# acne_detection
acne_detection

ran in python 3.6.8

## Requirements
* tensorflow == 1.13.1

## Training

```shell
export PYTHONPATH=$PYTHONPATH:/PATH_TO_THE_PROJECT/slim/

nohup python3.6 object_detection/model_main.py --pipeline_config_path=faster_rcnn_resnet101_coco.config --model_dir=./saved_models/ --num_train_steps=20000 --num_eval_steps=2000 --alsologtostderr > acne_train.log &
```

## Exporting the model

```shell
python3.6 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix ./saved_models/model.ckpt-xxxxx --output_directory ./latest_models/
```
