import os
from PIL import Image
import webdataset as wds
import numpy as np
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset

import torch

class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
            "type":'image',
        }


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        
        image = self.vis_processor(image)
        image = np.array(image)

        caption = ann["caption"]
        # ======================== CHANGE BLOCK ========================
        h_caption = ann["h_caption"]

        return {
            "image": image,
            "text_input": caption,
            "h_caption": h_caption, 
            "image_id": self.img_ids[ann["image_id"]],
            "type":'image',
        }
        # ======================== END CHANGE BLOCK ========================

    def collater(self, batch):
        return {
          'image': torch.stack([torch.tensor(x['image']) for x in batch]),
          'text_input': [x['text_input'] for x in batch],
          'h_caption': [x['h_caption'] for x in batch],
          'image_id': [x['image_id'] for x in batch],
          'type': 'image',
        }
