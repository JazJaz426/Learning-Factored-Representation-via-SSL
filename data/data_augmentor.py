import albumentations as A
import random

class DataAugmentor():

    def __init__(self, transformation_types):

        transformation_dict = {
            'random_rotation': A.Rotate,
            'random_scaling': A.RandomScale,
            'random_horz_flip': A.HorizontalFlip,
            'random_vert_flip': A.VerticalFlip,
            'random_color_jitter': A.ColorJitter,
            'random_greyscale': A.ToGray,
            'random_masking': A.CoarseDropout
        }


        transformations = list(transformation_types.keys())
        random.shuffle(transformations)

        composed_transformation = []

        for t in transformations:
            transformation_args = transformation_types[t]

            for k in transformation_args:
                transformation_args[k] = tuple(transformation_args[k]) if type(transformation_args[k]) is list else transformation_args[k]

            composed_transformation.append(transformation_dict[t](**transformation_args))

        self.composed_transformation = A.Compose([transformation_dict[t](**transformation_types[t]) for t in transformations])

    def apply_transformation(self, image):
        #apply the composed transfomration based on the sequence transformations

        augmented = self.composed_transformation(image=image)
        
        return augmented['image']


