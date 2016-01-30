from __future__ import division
import math

import numpy

from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer

class BilinearRescale(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize an image to a fixed window size. Use bilinear interpolation with 4-relative nearest neighbors.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, image_shape, **kwargs):
        self.image_shape = image_shape
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(BilinearRescale, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        height, width = self.image_shape
        print(source.shape)
        if isinstance(source, numpy.ndarray) and source.ndim == 4:
            # Hardcoded assumption of (batch, channels, height, width).
            # This is what the fast Cython code supports.
            raise Exception
        
        elif all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in source):
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
        height, width = self.image_shape
        nb_channel = example.shape[0] #That line could be replace by something more elegant.
        
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        rescale_height, rescale_width = (image_height-1)/(height-1), (image_width-1)/(width-1)
        
        rescaled_image = np.zeros((nb_channel,height,width))
        #Might do a cleaner version eventually
        #print(example[0][0])
        #print(example.shape)
        
        for i,j in product(range(height), range(width)):
            x, y  = i*rescale_height, j * rescale_width
            x1 = np.array([math.floor(x), math.ceil(x)],dtype=np.intp)
            y1 = np.array([math.floor(y), math.ceil(y)],dtype=np.intp)
            dx,dy = x - x1[0], y - y1[0]
            del_x, del_y = x1[1]-x1[0], y1[1]-y1[0]
            
            #Stupid patching for float approximation induced problem
            if x1[1] == image_height:
                x1[1] -= 1
            if y1[1] == image_width:
                y1[1] -= 1
                
            if not x1[0] == x1[1] and not y1[0] == y1[1] :
                for c in range(nb_channel):
                    xy1,xy2 = example[c][np.ix_(x1,y1)]
                    x1y1,x2y1,x1y2,x2y2 = xy1[0],xy1[1],xy2[0],xy2[1]
                    rescaled_image[c,i,j] = (x2y1-x1y1)*dx/del_x + (x1y2 -x1y1) *dy/del_y + (x1y1 + x2y2 - x2y1 - x1y2) *dx/del_x*dy/del_y + x1y1
            elif x1[0] == x1[1] and y1[0] == y1[1]:
                rescaled_image[:,i,j] = example[:,i,j]
            else:
                if y1[0] == y1[1]:
                    for c in range(nb_channel):
                        x1y,x2y = example[c][np.ix_(x1,y1)]
                        x1y,x2y = x1y[0], x1y[1]
                        rescaled_image[c,i,j] = (x2y-x1y)*dx/del_x + x1y
                else:
                    for c in range(nb_channel):
                        xy1,xy2 = example[c][np.ix_(x1,y1)]
                        xy1,xy2 = xy1[0], xy1[1]
                        rescaled_image[c,i,j] = (xy2 - xy1) * dy/del_y + xy1
            
        return rescaled_image
    

