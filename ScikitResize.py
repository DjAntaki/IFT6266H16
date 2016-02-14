from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
from skimage.transform import resize
import numpy as np

class ScikitResize(SourcewiseTransformer, ExpectsAxisLabels):
    def __init__(self, data_stream, image_shape, **kwargs):
        self.image_shape = image_shape
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ScikitResize, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        print("a")
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, np.ndarray) and source.ndim == 4:
            # Not yet supported(batch, channels, height, width).
            raise Exception
#            for x in range(source.shape[0]):
#                self.transform_source_example(im,source_name)
        
        elif all(isinstance(b, np.ndarray) and b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)

        if not isinstance(example, np.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        
        num_channel = example.shape[0]
    
        resized = np.zeros((num_channel,self.image_shape[0],self.image_shape[1]))
        for x in range(num_channel):
            resized[x] = resize(example[x], self.image_shape)
        return resized
