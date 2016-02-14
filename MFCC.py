#wrapping python-speech-feature librairy with SourwiseTransformer and ExpectsAxisLabels from fuel.transformers



import numpy

from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer

class MFCC(SourcewiseTransformer, ExpectsAxisLabels):
    """  """
    def __init__(self, data_stream, winlen=0.025,winstep=0.01,numcep=13,
          nfilt=26,nfft=512, **kwargs):
        self.winlen = winlen
        self.winstep = winstep
        self.nfilt=nfilt
        self.nfft=nfft
        self.numcep=numcep
        
#        self.lowfreq=0
#        self.highfreq=None
#        self.preemph=0.97,
#        self.ceplifter=22
#        self.appendEnergy=True
        
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(MFCC, self).__init__(data_stream, **kwargs)

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
        self.verify_axis_labels(('batch', 'time', 'features'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        
        print(example.shape)
        batch, time, features = example.shape 
        
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
                
        return features.mfcc(signal, winlen = self.winlen,
        winstep = self.winstep, nfilt=self.nfilt, nfft=self.nfft)
#        for x, d in enumerate(data):    
        for i in range(nbcapteurs):

            #signal = d[1+i*buffer_size:1+i*buffer_size + length]# I do not get a fix number of frame if I do not use the zero-padding.
            #print(length)
            signal = d[1+i*buffer_size: 1+(i+1)*buffer_size]
            mfcc_features[x,i,:,:] = features.mfcc(signal, winlen = self.winlen,
        winstep = self.winstep, nfilt=self.nfilt, nfft=self.nfft)
        
        return mfcc_features        
    

