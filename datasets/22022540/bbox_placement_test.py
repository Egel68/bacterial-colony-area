import detectron2, cv2, os, json
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

register_coco_instances('test', {}, 'annot_COCO.json', '')
dataset_dicts = DatasetCatalog.get('test')
metadata = MetadataCatalog.get('test')
for d in dataset_dicts:
  imageName = d['file_name']
  img = cv2.imread(imageName)
  visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.9)
  vis = visualizer.draw_dataset_dict(d)
  cv2.imwrite(imageName.replace('.jpg', '_res.jpg'), vis.get_image()[:, :, ::-1])
