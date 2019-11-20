from PIL import Image, ImageDraw
import os
import torchvision.datasets as dset
from torch import FloatTensor, LongTensor, ByteTensor
from torchvision.transforms import ToPILImage

def to_xy(bbox):
    coors = [(bbox[0], bbox[1]), (bbox[2], bbox[1]),
             (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    return coors

def visualize_boxes(sample, max_box=10):
    im = ToPILImage()(sample[0])
    draw = ImageDraw.Draw(im)
    k = 0
    for bb in sample[1]['boxes']:
        k += 1
        draw.polygon(to_xy(bb), outline=(255, 0, 0))
        if (sample[1]['labels'][k - 1].item() == 1):
            label = 'Person'
        else:
            label = 'Car'
        draw.text((bb[0], bb[1]), label, (0, 255, 0))
        if (k == max_box):
            break
    del draw
    return im
 
table = [i/256 for i in range(65536)]

class CocoDetection_(dset.CocoDetection):
    def __getitem__(self, index):
        """
          Args:
              index (int): Index

          Returns:
              tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)
        
        ann = {k: [dic[k] for dic in ann] for k in ann[0]}
            
        bbox = []
        for bb in ann['bbox']:
            bbox.append([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])
        
        target = {}
        target['area'] = FloatTensor(ann['area'])
        target['boxes'] = FloatTensor(bbox)
        target['labels'] = LongTensor(ann['category_id'])
        target['image_id'] = LongTensor([img_id])
        target['iscrowd'] = ByteTensor(ann['iscrowd'])

        path = coco.loadImgs(img_id)[0]['file_name']
        
        im = Image.open(os.path.join(self.root, path))
        im = im.point(table, 'L')
        im = im.convert('RGB')
        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

class Pretrain(dset.CocoDetection):
    def __getitem__(self, index):
        """
          Args:
              index (int): Index

          Returns:
              tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)
        
        ann = {k: [dic[k] for dic in ann] for k in ann[0]}
            
        bbox = []
        for bb in ann['bbox']:
            bbox.append([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])
        
        target = {}
        target['area'] = FloatTensor(ann['area'])
        target['boxes'] = FloatTensor(bbox)
        target['labels'] = LongTensor(ann['category_id'])
        target['image_id'] = LongTensor([img_id])
        target['iscrowd'] = ByteTensor(ann['iscrowd'])

        path = coco.loadImgs(img_id)[0]['file_name']
        
        im = Image.open(os.path.join(self.root, path))
        im = im.convert('RGB')
        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target    
    
class Test(dset.CocoDetection):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
         
        path = coco.loadImgs(img_id)[0]['name']
        
        im = Image.open(os.path.join(self.root, path))
        im = im.point(table, 'L')
        im = im.convert('RGB')
        if self.transforms is not None:
            im, target = self.transforms(im, img_id)       
        return im, img_id
